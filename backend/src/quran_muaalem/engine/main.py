import litserve as ls

from .serve import QuranMuaalemAPI
from .settings import EngineSettings


def main():
    engine_settings = EngineSettings()

    # Instantiate the API with engine_settings
    api = QuranMuaalemAPI(
        model_name_or_path=engine_settings.model_name_or_path,
        dtype=engine_settings.torch_dtype,
        max_audio_seconds=engine_settings.max_audio_seconds,
        max_batch_size=engine_settings.max_batch_size,
        batch_timeout=engine_settings.batch_timeout,
    )

    # Create the LitServer, passing batching parameters to avoid the warning
    server = ls.LitServer(
        api,
        accelerator=engine_settings.accelerator,
        devices=engine_settings.devices,
        timeout=engine_settings.timeout,
        workers_per_device=engine_settings.workers_per_device,
    )

    # Run the server
    server.run(port=engine_settings.port)


if __name__ == "__main__":
    main()
