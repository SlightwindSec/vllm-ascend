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

# 本算子有两层结构：
# 1. 底层 kernel: ApplyTopKTopPWithSortedCustom (处理已排序的输入)
# 2. 顶层 aclnn API: aclnnApplyTopKTopPCustom (先排序再调用底层 kernel)
#
# 因为我们手动提供了 aclnn 实现（包含排序逻辑），所以不能使用 op_host_aclnn 目标
# （op_host_aclnn 会触发自动生成 aclnn 文件，与我们手动提供的文件冲突）
# 
# 将 def 文件直接添加到 opsproto 目标，避免触发自动生成

if (BUILD_OPEN_PROJECT)
    # 算子定义和形状推导 - 添加到 opsproto
    target_sources(opsproto PRIVATE
            apply_top_k_top_p_custom_def.cpp
            apply_top_k_top_p_custom_infershape.cpp
    )

    # 手动提供的 aclnn 实现和 l0op 实现 - 添加到 opapi
    target_sources(opapi PRIVATE
            apply_top_k_top_p_custom.cpp
            aclnn_apply_top_k_top_p_custom.cpp
    )

    # Tiling 实现
    target_sources(optiling PRIVATE
            apply_top_k_top_p_custom_tiling.cpp
    )

    target_include_directories(optiling PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}
    )

    file(GLOB _GMM_Aclnn_header "${CMAKE_CURRENT_SOURCE_DIR}/aclnn_apply_top_k_top_p_custom.h")

    install(FILES ${_GMM_Aclnn_header}
            DESTINATION ${ACLNN_INC_INSTALL_DIR} OPTIONAL
    )
endif ()

if (NOT BUILD_OPEN_PROJECT)
    target_sources(aclnn_ops_train PRIVATE
            apply_top_k_top_p_custom_def.cpp
            apply_top_k_top_p_custom.cpp
            aclnn_apply_top_k_top_p_custom.cpp
    )

    target_sources(aclnn_ops_infer PRIVATE
            apply_top_k_top_p_custom_def.cpp
            apply_top_k_top_p_custom.cpp
            aclnn_apply_top_k_top_p_custom.cpp
    )
endif ()
