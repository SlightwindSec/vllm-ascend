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
 * \file apply_top_k_top_p_with_sorted_custom_infershape.cpp
 * \brief ApplyTopKTopPWithSortedCustom 算子的形状推导和数据类型推导实现
 */
#include "register/op_impl_registry.h"
#include "log/ops_log.h"

using namespace ge;
namespace ops {

// 输入索引
static constexpr int64_t SORTED_VALUE_INDEX = 0;
static constexpr int64_t SORTED_INDICES_INDEX = 1;
static constexpr int64_t P_INDEX = 2;
static constexpr int64_t K_INDEX = 3;

// 输出索引
static constexpr int64_t OUT_INDEX = 0;

/**
 * @brief ApplyTopKTopPWithSortedCustom 算子的形状推导函数
 * 输出形状与 sorted_value 输入形状相同
 */
static ge::graphStatus InferShape4ApplyTopKTopPWithSortedCustom(gert::InferShapeContext* context)
{
    OPS_LOG_D(context->GetNodeName(), "Begin to do ApplyTopKTopPWithSortedCustomInfershape.");
    
    // 获取输入 sorted_value 的形状
    const gert::Shape* sortedValueShape = context->GetInputShape(SORTED_VALUE_INDEX);
    if (sortedValueShape == nullptr) {
        OPS_LOG_E(context->GetNodeName(), "Failed to get sorted_value shape.");
        return ge::GRAPH_FAILED;
    }
    
    // 获取输出形状指针
    gert::Shape* outShape = context->GetOutputShape(OUT_INDEX);
    if (outShape == nullptr) {
        OPS_LOG_E(context->GetNodeName(), "Failed to get output shape.");
        return ge::GRAPH_FAILED;
    }
    
    // 输出形状与 sorted_value 输入形状相同
    size_t dimNum = sortedValueShape->GetDimNum();
    outShape->SetDimNum(dimNum);
    for (size_t i = 0; i < dimNum; ++i) {
        outShape->SetDim(i, sortedValueShape->GetDim(i));
    }
    
    OPS_LOG_D(context->GetNodeName(), "End to do ApplyTopKTopPWithSortedCustomInfershape.");
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief ApplyTopKTopPWithSortedCustom 算子的数据类型推导函数
 * 输出数据类型与 sorted_value 输入数据类型相同
 */
static ge::graphStatus InferDataType4ApplyTopKTopPWithSortedCustom(gert::InferDataTypeContext* context)
{
    OPS_LOG_D(context->GetNodeName(), "Begin to do ApplyTopKTopPWithSortedCustomInferDataType.");
    
    // 输出数据类型与 sorted_value 输入数据类型相同
    auto sortedValueDtype = context->GetInputDataType(SORTED_VALUE_INDEX);
    context->SetOutputDataType(OUT_INDEX, sortedValueDtype);
    
    OPS_LOG_D(context->GetNodeName(), "End to do ApplyTopKTopPWithSortedCustomInferDataType.");
    return ge::GRAPH_SUCCESS;
}

// 注册算子的形状推导和数据类型推导实现
IMPL_OP_INFERSHAPE(ApplyTopKTopPWithSortedCustom)
    .InferShape(InferShape4ApplyTopKTopPWithSortedCustom)
    .InferDataType(InferDataType4ApplyTopKTopPWithSortedCustom);

}  // namespace ops
