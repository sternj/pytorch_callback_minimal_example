#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <cupti.h>
#include <cupti_activity.h>
#include <ATen/record_function.h>
#include "libcbtest.hpp"

#include <set>

static thread_local std::unordered_set<at::CallbackHandle> active_bwd_ops;

std::unique_ptr<at::ObserverContext> nullCb(const at::RecordFunction &fn)
{
    return nullptr;
}

uint64_t begin_fwd_op(int marker)
{
    auto cb_handle = at::addThreadLocalCallback(at::RecordFunctionCallback(nullCb, nullptr).scopes({at::RecordScope::BACKWARD_FUNCTION}));
    return cb_handle;
}

void begin_bwd_op(uint64_t bwd_cbid)
{
    active_bwd_ops.insert(static_cast<at::CallbackHandle>(bwd_cbid));
}

void end_bwd_op(uint64_t bwd_cbid)
{
    auto tls = at::get_record_function_tls_();
    assert(tls.sorted_tls_callbacks_.size() == active_bwd_ops.size());
    at::removeCallback(static_cast<at::CallbackHandle>(bwd_cbid));
    // NOTE: I don't know if TLS was updated, so I'm just re-retrieving
    auto tls2 = at::get_record_function_tls_();
    active_bwd_ops.erase(static_cast<at::CallbackHandle>(bwd_cbid));
    assert(tls2.sorted_tls_callbacks_.size() == active_bwd_ops.size());
}

PYBIND11_MODULE(_C, m)
{
    m.def("_begin_fwd_op", &begin_fwd_op, "Begin forward");
    m.def("_begin_bwd_op", &begin_bwd_op, "Begin backward op");
    m.def("_end_bwd_op", &end_bwd_op, "End backward op");
}