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

# 将 def 文件添加到 op_host_aclnnExc，用于生成 proto 等文件
# 使用 Exc（Exclude）是因为已经有手动编写的 aclnn 实现，不需要自动生成
target_sources(op_host_aclnnExc PRIVATE
        apply_top_k_top_p_with_sorted_def.cpp
)

target_sources(opapi PRIVATE
        apply_top_k_top_p_with_sorted.cpp
        aclnn_apply_top_k_top_p.cpp
)

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

target_sources(optiling PRIVATE
        apply_top_k_top_p_with_sorted_tiling.cpp
)

target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_sources(opsproto PRIVATE
        apply_top_k_top_p_with_sorted_infershape.cpp
)

file(GLOB _GMM_Aclnn_header "${CMAKE_CURRENT_SOURCE_DIR}/aclnn_apply_top_k_top_p.h")

install(FILES ${_GMM_Aclnn_header}
        DESTINATION ${ACLNN_INC_INSTALL_DIR} OPTIONAL
)