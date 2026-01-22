/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file apply_top_k_top_p_with_sorted_proto.cpp
 * \brief ApplyTopKTopPWithSortedCustom
 */

#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>

using namespace ge;

namespace ops {
constexpr size_t SORTED_VALUE_INPUT_INDEX = 0;

/**
 * @brief ApplyTopKTopPWithSortedCustom 算子的形状推理函数
 * @param context 推理上下文
 * @return 成功返回 GRAPH_SUCCESS，失败返回 GRAPH_FAILED
 *
 * 输出形状与输入 sorted_value 相同
 */
static ge::graphStatus InferShapeApplyTopKTopPWithSortedCustom(gert::InferShapeContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    // 获取输入 sorted_value 的形状
    const gert::Shape *sortedValueShape = context->GetInputShape(SORTED_VALUE_INPUT_INDEX);
    if (sortedValueShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    // 获取输出形状指针
    gert::Shape *outShape = context->GetOutputShape(0);
    if (outShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    // 输出形状与 sorted_value 输入形状相同
    *outShape = *sortedValueShape;
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief ApplyTopKTopPWithSortedCustom 算子的数据类型推理函数
 * @param context 推理上下文
 * @return 成功返回 GRAPH_SUCCESS，失败返回 GRAPH_FAILED
 *
 * 输出数据类型与输入 sorted_value 相同
 */
static ge::graphStatus InferDataTypeApplyTopKTopPWithSortedCustom(gert::InferDataTypeContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    // 获取输入 sorted_value 的数据类型
    const auto inputDataType = context->GetInputDataType(SORTED_VALUE_INPUT_INDEX);
    // 设置输出数据类型与输入相同
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}

// 注册 ApplyTopKTopPWithSortedCustom 算子的推理实现
IMPL_OP(ApplyTopKTopPWithSortedCustom)
    .InferShape(InferShapeApplyTopKTopPWithSortedCustom)
    .InferDataType(InferDataTypeApplyTopKTopPWithSortedCustom);
} // namespace ops
