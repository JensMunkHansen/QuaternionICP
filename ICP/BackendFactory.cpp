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

std::unique_ptr<IntersectionBackend> createIntersectionBackend()
{
#if ICP_USE_EMBREE
    return std::make_unique<EmbreeBackend>();
#elif ICP_USE_GRIDSEARCH
    return std::make_unique<GridSearchBackend>();
#else
    return nullptr;
#endif
}

}  // namespace ICP
