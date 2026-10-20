/* Exercise the real VA entrypoints without loading libcuda or opening a GPU.
 * Only the CUDA/NVDEC boundary is replaced, to inject context-stack failures.
 * Including the implementation keeps its private entrypoints private. */
#include "../src/vabackend.c"

/* The driver's constructor normally loads CUDA before main. Block dlopen for
 * this executable as well, so even a machine with NVIDIA installed stays on
 * the CPU and the test cannot initialize the real driver. */
void *__wrap_dlopen(const char *name, int flags);
void *__wrap_dlopen(const char *name, int flags) {
    return NULL;
}

static bool failPush;
static bool failPop;
static unsigned int abortCalls;

static void fakeAbortPicture(NVContext *context) {
    abortCalls++;
    context->renderTarget->decodeFailed = true;
    setSurfaceResolving(context->renderTarget, false);
}

static CUresult CUDAAPI fakePush(CUcontext context) {
    return failPush ? (CUresult) 201 /* CUDA_ERROR_INVALID_CONTEXT */ : CUDA_SUCCESS;
}

static CUresult CUDAAPI fakePop(CUcontext *context) {
    return failPop ? (CUresult) 201 /* CUDA_ERROR_INVALID_CONTEXT */ : CUDA_SUCCESS;
}

static CUresult CUDAAPI fakeErrorString(CUresult result, const char **message) {
    *message = "injected context failure";
    return CUDA_SUCCESS;
}

static CUresult CUDAAPI fakeDecode(CUvideodecoder decoder, CUVIDPICPARAMS *picture) {
    return CUDA_SUCCESS;
}

static void require(bool condition, const char *message) {
    if (!condition) {
        fprintf(stderr, "FAIL: %s\n", message);
        cu = NULL;
        cv = NULL;
        exit(1);
    }
}

int main(int argc, char **argv) {
    if ((argc != 2 && argc != 3) ||
        (strcmp(argv[1], "push") && strcmp(argv[1], "pop") && strcmp(argv[1], "success")) ||
        (argc == 3 && strcmp(argv[2], "distinct-display") && strcmp(argv[2], "codec-abort"))) {
        return 2;
    }
    failPush = !strcmp(argv[1], "push");
    failPop = !strcmp(argv[1], "pop");

    CudaFunctions fakeCuda = {
        .cuCtxPushCurrent = fakePush,
        .cuCtxPopCurrent = fakePop,
        .cuGetErrorString = fakeErrorString,
    };
    CuvidFunctions fakeCuvid = { .cuvidDecodePicture = fakeDecode };
    cu = &fakeCuda;
    cv = &fakeCuvid;

    NVDriver drv = { .cu = cu, .cv = cv };
    struct VADriverContext va = { .pDriverData = &drv };
    pthread_mutexattr_t attr;
    pthread_mutexattr_init(&attr);
    pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE);
    pthread_mutex_init(&drv.objectCreationMutex, &attr);
    pthread_mutexattr_destroy(&attr);

    Object contextObject = allocateObject(&drv, OBJECT_TYPE_CONTEXT, sizeof(NVContext));
    NVContext *context = contextObject->obj;
    const bool codecAbort = argc == 3 && !strcmp(argv[2], "codec-abort");
    const bool distinctDisplay = argc == 3 && !strcmp(argv[2], "distinct-display");
    // H.264/VP9/AV1 have no abortPicture callback; VC1 supplies one.
    const NVCodec codec = { .abortPicture = codecAbort ? fakeAbortPicture : NULL };
    context->drv = &drv;
    context->codec = &codec;
    context->entrypoint = VAEntrypointVLD;
    context->decoder = (CUvideodecoder) (uintptr_t) 1;
    context->surfaceCount = 4;
    pthread_mutex_init(&context->resolveMutex, NULL);
    pthread_cond_init(&context->resolveCondition, NULL);
    pthread_cond_init(&context->videoProcCondition, NULL);

    Object surfaceObject = allocateObject(&drv, OBJECT_TYPE_SURFACE, sizeof(NVSurface));
    NVSurface *surface = surfaceObject->obj;
    surface->pictureIdx = -1;
    surface->contextId = VA_INVALID_ID;
    pthread_mutex_init(&surface->mutex, NULL);
    pthread_cond_init(&surface->cond, NULL);
    BackingImage image = { .syncInitialized = true };
    pthread_mutex_init(&image.mutex, NULL);
    pthread_cond_init(&image.cond, NULL);
    surface->backingImage = &image;

    require(nvBeginPicture(&va, contextObject->id, surfaceObject->id) == VA_STATUS_SUCCESS,
            "BeginPicture must succeed");
    require(surface->resolving && image.resolving, "BeginPicture marks the surface busy");
    // AV1 film grain can select a distinct display target during RenderPicture.
    // It is only marked for this picture after the context-stack error paths;
    // its existing work must remain owned by the earlier submission.
    NVSurface display = { .resolving = 1 };
    pthread_mutex_init(&display.mutex, NULL);
    pthread_cond_init(&display.cond, NULL);
    if (distinctDisplay) {
        context->displayTarget = &display;
    }
    VAStatus status = nvEndPicture(&va, contextObject->id);
    require(context->activeDecodeCalls == 0, "EndPicture must release its context reference");

    if (failPush || failPop) {
        require(status == VA_STATUS_ERROR_OPERATION_FAILED, "preserve the CUDA error status");
        if (distinctDisplay) {
            require(display.resolving && !display.decodeFailed,
                    "do not release a distinct display target's earlier submission");
        }
        puts("Syncing the failed picture: this must return without a resolver thread.");
        fflush(stdout);
        require(nvSyncSurface(&va, surfaceObject->id) == VA_STATUS_SUCCESS,
                "SyncSurface must finish after the failed submission");
        require(!surface->resolving && !image.resolving, "release surface and backing image waiters");
        require(surface->decodeFailed, "do not publish the failed picture as valid");
        require(context->surfaceQueueWriteIdx == context->surfaceQueueReadIdx,
                "failed submissions must not enter the resolver queue");
        require(abortCalls == (codecAbort ? 1u : 0u), "preserve codec-specific abort handling");
    } else {
        require(status == VA_STATUS_SUCCESS, "successful submissions still succeed");
        require(surface->resolving && image.resolving, "successful submissions stay busy until resolved");
        require(!surface->decodeFailed, "do not mark a successful picture failed");
        require(context->surfaceQueueWriteIdx != context->surfaceQueueReadIdx &&
                context->surfaceQueue[context->surfaceQueueReadIdx] == surface,
                "successful submissions must reach the resolver queue");
    }

    free(context->surfaceQueue);
    pthread_cond_destroy(&context->videoProcCondition);
    pthread_cond_destroy(&context->resolveCondition);
    pthread_mutex_destroy(&context->resolveMutex);
    pthread_cond_destroy(&surface->cond);
    pthread_mutex_destroy(&surface->mutex);
    pthread_cond_destroy(&image.cond);
    pthread_mutex_destroy(&image.mutex);
    pthread_cond_destroy(&display.cond);
    pthread_mutex_destroy(&display.mutex);
    deleteObject(&drv, surfaceObject->id);
    deleteObject(&drv, contextObject->id);
    free(drv.objects.buf);
    pthread_mutex_destroy(&drv.objectCreationMutex);
    // The constructor's destructor must not free our stack-allocated tables.
    cu = NULL;
    cv = NULL;
    puts("PASS");
    return 0;
}
