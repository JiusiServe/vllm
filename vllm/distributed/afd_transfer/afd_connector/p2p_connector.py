# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import re
from datetime import timedelta
from typing import Any, Optional

import torch
from torch.distributed.distributed_c10d import  _update_default_pg, _get_default_group
from .base import AFDConnectorBase
from .metadata import AFDConnectorMetadata
from vllm.distributed.parallel_state import init_afd_process_group, init_model_parallel_group, _split_tensor_dict, TensorMetadata, GroupCoordinator
from vllm.sequence import IntermediateTensors
from vllm.logger import init_logger
from vllm.config import VllmConfig
logger = init_logger(__name__)

class DefaultProcessGroupSwitcher:

    def __init__(self, default_group, new_default_group):
        self.default_group = default_group
        self.new_default_group = new_default_group

    def __enter__(self):
        _update_default_pg(self.new_default_group)

    def __exit__(self, exc_type, exc_value, traceback):
        _update_default_pg(self.default_group)


class P2PAFDConnector(AFDConnectorBase):
    def __init__(self,
        rank: int,
        local_rank: int,
        config: "VllmConfig",
) -> None:
        self.rank = rank
        self.local_rank = local_rank
        self._initialized = False
        self.config = config

    def close(self) -> None:
        """Close the connector and release resources."""
        # destroy process group
        pass

    def init_afd_connector(self) -> None:
        """Initialize the AFD connector."""
        afd_size = self.config.afd_config.afd_extra_config.get("afd_size")
        role = self.config.afd_config.afd_role
        attn_size, ffn_size = map(
            int,
            re.match(r"(\d+)\D+(\d+)", afd_size).groups())
        #ffn_ranks = [i for i in range(ffn_size, ffn_size + attn_size)]
        #attn_ranks = [i for i in range(attn_size)]
        world_rank = self.rank if role == "attention" else self.rank + attn_size

        logger.info(
            f"world_size = {ffn_size + attn_size}, world_rank = {world_rank}")
        afd_pg = init_afd_process_group(
            backend="nccl",
            init_method=f"tcp://127.0.0.1:29500",
            world_size=ffn_size + attn_size,
            rank=world_rank,
            group_name="afd",
            timeout=timedelta(minutes=2),
        )
        ffn_ranks = [i for i in range(ffn_size, ffn_size + attn_size)]
        attn_ranks = [i for i in range(attn_size)]

        default_pg_switcher = DefaultProcessGroupSwitcher(
            _get_default_group(), afd_pg)
        with default_pg_switcher:
            sub_group_ranks = []
            for i in range(len(ffn_ranks)):
                ranks = list([attn_ranks[i], ffn_ranks[i]])
                sub_group_ranks.append(ranks)
            # Create two independent groups:
            # a2e_group: for attention -> expert/ffn communication (send_attn, recv_attn)
            # e2a_group: for expert/ffn -> attention communication (send_ffn, recv_ffn)
            # The communication domain (rank range) is the same, but different group_name
            # creates independent groups
            self.a2e_group = init_model_parallel_group(sub_group_ranks,
                                                 self.local_rank,
                                                 backend="nccl",
                                                 group_name="a2e")
            self.e2a_group = init_model_parallel_group(sub_group_ranks,
                                                 self.local_rank,
                                                 backend="nccl",
                                                 group_name="e2a")

        logger.info("p2p connector initialized")

        self._initialized = True

    def is_initialized(self) -> bool:
        """Check if the connector is initialized and ready to use.
        
        Returns:
            bool: True if the connector is initialized, False otherwise.
        """
        return self._initialized

    def _send_tensor_dict_async(
        self,
        tensor_dict: dict[str, torch.Tensor],
        dst: int,
        process_group: GroupCoordinator,
    ) -> list:
        """Asynchronously send a tensor dictionary.
        
        Args:
            tensor_dict: The tensor dictionary to send
            dst: Destination rank (local rank)
            process_group: The process group to use for communication
            
        Returns:
            List of work objects that can be used to wait for operation completion
        """
        if not torch.distributed.is_initialized() or process_group.world_size == 1:
            return []
        
        assert dst < process_group.world_size, f"Invalid dst rank ({dst})"
        
        # Split tensor dictionary into metadata and tensor list
        metadata_list, tensor_list = _split_tensor_dict(tensor_dict)
        
        # Send metadata first (synchronously, as metadata is small and on CPU)
        process_group.send_object(metadata_list, dst=dst)
        
        # Asynchronously send each tensor
        work_list = []
        group = process_group.device_group
        metadata_group = process_group.cpu_group
        
        for tensor in tensor_list:
            if tensor.numel() == 0:
                # Skip empty tensors
                continue
            
            if tensor.is_cpu:
                # CPU tensor uses metadata_group
                work = torch.distributed.isend(
                    tensor, dst=process_group.ranks[dst], group=metadata_group
                )
            else:
                # GPU tensor uses device_group
                work = torch.distributed.isend(
                    tensor, dst=process_group.ranks[dst], group=group
                )
            work_list.append(work)
        
        return work_list

    def _recv_tensor_dict_async(
        self,
        src: int,
        process_group: GroupCoordinator,
        all_gather_group: Optional["GroupCoordinator"] = None,
    ) -> tuple[dict[str, torch.Tensor | Any], list]:
        """Asynchronously receive a tensor dictionary.
        
        Args:
            src: Source rank (local rank)
            process_group: The process group to use for communication
            all_gather_group: Group for all-gather optimization
            
        Returns:
            tuple: (tensor_dict, work_list) - tensor dictionary and work object list
        """
        if not torch.distributed.is_initialized() or process_group.world_size == 1:
            return {}, []
        
        assert src < process_group.world_size, f"Invalid src rank ({src})"
        
        # Receive metadata first synchronously (need to know tensor shape and type)
        recv_metadata_list = process_group.recv_object(src=src)
        
        # Create empty tensor dictionary and work list
        tensor_dict: dict[str, Any] = {}
        work_list = []
        group = process_group.device_group
        metadata_group = process_group.cpu_group
        
        for key, value in recv_metadata_list:
            if isinstance(value, TensorMetadata):
                # Create empty tensor from metadata
                tensor = torch.empty(value.size, dtype=value.dtype, device=value.device)
                
                if tensor.numel() == 0:
                    # Skip empty tensors
                    tensor_dict[key] = tensor
                    continue
                
                # Asynchronously receive tensor
                if tensor.is_cpu:
                    # CPU tensor uses metadata_group
                    work = torch.distributed.irecv(
                        tensor, src=process_group.ranks[src], group=metadata_group
                    )
                else:
                    # GPU tensor uses device_group
                    work = torch.distributed.irecv(
                        tensor, src=process_group.ranks[src], group=group
                    )
                work_list.append(work)
                tensor_dict[key] = tensor
            else:
                # Non-tensor values are added directly
                tensor_dict[key] = value
        
        return tensor_dict, work_list

    def send_attn_output(
        self, hidden_states: torch.Tensor, metadata: AFDConnectorMetadata
    ):
        """
        This method will be called by the ATTN side.


        * To send the intermediate tensors generated by ATTN instances to FFN.
        """

        intermediate_tensors = IntermediateTensors(
            {
                "hidden_states": hidden_states,
            }
        )
        try:
            # Use async send instead of sync send
            # Use a2e_group for attention -> expert/ffn communication
            torch.cuda.current_stream().synchronize()
            dst = (self.a2e_group.rank_in_group + 1) % self.a2e_group.world_size
            work_list = self._send_tensor_dict_async(
                intermediate_tensors.tensors,
                dst=dst,
                process_group=self.a2e_group,
            )
            # work_list can be used for waiting later if we need to ensure send completion
            # Here we don't wait, letting the send proceed asynchronously in the background
            self.a2e_group.send_object(metadata, dst)
            if metadata is not None:
                metadata.send_handle_list = work_list
        except Exception as e:
            raise RuntimeError(f"Communication error: {e}")

    def recv_attn_output(self) -> IntermediateTensors:
        """
        This method will be called by the FFN side.


        * To receive the intermediate tensors from ATTN.
        * And (Maybe) dispatch them from the receiver to other GPUs.
        """
        # Use a2e_group for attention -> expert/ffn communication
        src = (self.a2e_group.rank_in_group - 1) % self.a2e_group.world_size
        # Use async receive for tensor_dict
        intermediate_tensors, work_list = self._recv_tensor_dict_async(
            src=src,
            process_group=self.a2e_group,
            all_gather_group=None,
        )
        # Asynchronously receive independent metadata
        metadata = self.a2e_group.recv_object(src)
        metadata.recv_handle_list = work_list
        return intermediate_tensors["hidden_states"], metadata

    # -------------------------------------------------------------------------
    #                                attn <- ffn
    # -------------------------------------------------------------------------
    def send_ffn_output(
        self, hidden_states: torch.Tensor, metadata: AFDConnectorMetadata
    ):
        """
        This method will be called by the FFN side.


        * To send the intermediate tensors generated by FFN instances back to
            the sender (this should be the same GPU as it comes from)
        """
        intermediate_tensors = IntermediateTensors(
            {
                "hidden_states": hidden_states,
            }
        )
        # Use async send instead of sync send
        # Use e2a_group for expert/ffn -> attention communication
        torch.cuda.current_stream().synchronize()
        dst = (self.e2a_group.rank_in_group + 1) % self.e2a_group.world_size
        work_list = self._send_tensor_dict_async(
            intermediate_tensors.tensors,
            dst=dst,
            process_group=self.e2a_group,
        )
        # work_list can be used for waiting later if we need to ensure send completion
        # Here we don't wait, letting the send proceed asynchronously in the background
        self.e2a_group.send_object(metadata, dst)
        if metadata is not None:
            metadata.send_handle_list = work_list

    def recv_ffn_output(self) -> torch.Tensor:
        """
        This method will be called by the ATTN side.


        * To receive the MOE output intermediate tensors.
        * And (Maybe) dispatch them from the receiver to other GPUs.
            (this should be the same GPU as it comes from)
        """
        # Use e2a_group for expert/ffn -> attention communication
        src = (self.e2a_group.rank_in_group - 1) % self.e2a_group.world_size
        # Use async receive for tensor_dict
        intermediate_tensors, work_list = self._recv_tensor_dict_async(
            src=src,
            process_group=self.e2a_group,
            all_gather_group=None,
        )
        # Asynchronously receive independent metadata
        metadata = self.e2a_group.recv_object(src)
        # Wait for tensor receive completion (because we need to use data immediately)
        metadata.recv_handle_list = work_list
        return intermediate_tensors["hidden_states"], metadata
