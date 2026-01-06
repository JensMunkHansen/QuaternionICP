// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Jens Munk Hansen

// Standard C++ headers
#include <atomic>
#include <stdexcept>

// Internal headers
#include <ICP/Config.h>
#include <ICP/IntersectionBackend.h>

#if ICP_USE_EMBREE
#include <ICP/EmbreeBackend.h>
#endif

#if ICP_USE_GRIDSEARCH
#include <ICP/GridSearchBackend.h>
#endif

namespace ICP
{

// Default backend type (GridSearch preferred when available)
static std::atomic<IntersectionBackendType> g_defaultBackendType{IntersectionBackendType::Auto};

void setDefaultIntersectionBackend(IntersectionBackendType type)
{
    g_defaultBackendType.store(type, std::memory_order_relaxed);
}

IntersectionBackendType getDefaultIntersectionBackend()
{
    return g_defaultBackendType.load(std::memory_order_relaxed);
}

bool isBackendAvailable(IntersectionBackendType type)
{
    switch (type)
    {
        case IntersectionBackendType::Auto:
            return true;  // Auto always succeeds (falls back)
        case IntersectionBackendType::GridSearch:
#if ICP_USE_GRIDSEARCH
            return true;
#else
            return false;
#endif
        case IntersectionBackendType::Embree:
#if ICP_USE_EMBREE
            return true;
#else
            return false;
#endif
        default:
            return false;
    }
}

std::unique_ptr<IntersectionBackend> createIntersectionBackend(IntersectionBackendType type)
{
    // Resolve Auto to the actual default
    if (type == IntersectionBackendType::Auto)
    {
        type = g_defaultBackendType.load(std::memory_order_relaxed);

        // If still Auto, use compile-time priority: GridSearch > Embree
        if (type == IntersectionBackendType::Auto)
        {
#if ICP_USE_GRIDSEARCH
            type = IntersectionBackendType::GridSearch;
#elif ICP_USE_EMBREE
            type = IntersectionBackendType::Embree;
#endif
        }
    }

    // Create the requested backend
    switch (type)
    {
        case IntersectionBackendType::GridSearch:
#if ICP_USE_GRIDSEARCH
            return std::make_unique<GridSearchBackend>();
#else
            throw std::runtime_error("GridSearch backend not available (compile with -DUSE_GRIDSEARCH=ON)");
#endif

        case IntersectionBackendType::Embree:
#if ICP_USE_EMBREE
            return std::make_unique<EmbreeBackend>();
#else
            throw std::runtime_error("Embree backend not available (compile with -DUSE_EMBREE=ON)");
#endif

        default:
            return nullptr;
    }
}

}  // namespace ICP
