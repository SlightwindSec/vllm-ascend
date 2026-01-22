# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

add_ops_compile_options(
        OP_NAME ApplyTopKTopPWithSortedCustom
        OPTIONS --cce-auto-sync=on
                -Wno-deprecated-declarations
                -Werror
)

# 使用 op_host_aclnn 目标，参考 moe_gating_top_k 的配置
# 这样可以避免自动生成 proto 文件的问题
if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnn PRIVATE
            apply_top_k_top_p_with_sorted_def.cpp
    )

    target_sources(opapi PRIVATE
            apply_top_k_top_p_with_sorted.cpp
            aclnn_apply_top_k_top_p.cpp
    )

    target_sources(optiling PRIVATE
            apply_top_k_top_p_with_sorted_tiling.cpp
    )

    target_include_directories(optiling PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}
    )

    target_sources(opsproto PRIVATE
            apply_top_k_top_p_with_sorted_custom_infershape.cpp
    )

    file(GLOB _GMM_Aclnn_header "${CMAKE_CURRENT_SOURCE_DIR}/aclnn_apply_top_k_top_p.h")

    install(FILES ${_GMM_Aclnn_header}
            DESTINATION ${ACLNN_INC_INSTALL_DIR} OPTIONAL
    )
endif ()

if (NOT BUILD_OPEN_PROJECT)
    target_sources(aclnn_ops_train PRIVATE
            apply_top_k_top_p_with_sorted_def.cpp
            apply_top_k_top_p_with_sorted.cpp
            aclnn_apply_top_k_top_p.cpp
    )

    target_sources(aclnn_ops_infer PRIVATE
            apply_top_k_top_p_with_sorted_def.cpp
            apply_top_k_top_p_with_sorted.cpp
            aclnn_apply_top_k_top_p.cpp
    )
endif ()
