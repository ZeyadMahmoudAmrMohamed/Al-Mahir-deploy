import uvicorn

from .serve import app
from .settings import AppSettings


def main():
    app_settings = AppSettings()
    uvicorn.run(app, host=app_settings.host, port=app_settings.port)


if __name__ == "__main__":
    main()
