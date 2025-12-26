#include <ICP/CommonOptions.h>

int main(int argc, char** argv)
{
    ICP::CommonOptions opts;
    if (!ICP::parseArgs(argc, argv, opts, "SingleICP - Ray-projection ICP for two grids"))
    {
        return 1;
    }

    return 0;
}
