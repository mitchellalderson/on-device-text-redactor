import modal

app = modal.App("phi-firewall")

image = modal.Image.from_registry(
    "python:3.12-slim",
).pip_install("starlette")

dist = modal.Mount.from_local_dir("dist", remote_path="/app/dist")


@app.function(image=image, mounts=[dist], allow_concurrent_requests=100)
@modal.asgi_app()
def web():
    from starlette.staticfiles import StaticFiles
    from starlette.responses import FileResponse
    from starlette.routing import Mount, Route

    async def index(request):
        return FileResponse("/app/dist/index.html")

    return Starlette(
        routes=[
            Route("/", index),
            Mount("/", app=StaticFiles(directory="/app/dist", html=True), name="static"),
        ],
    )


try:
    from starlette.applications import Starlette
except ImportError:
    pass
