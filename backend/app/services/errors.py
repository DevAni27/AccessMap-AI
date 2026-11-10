"""One error type for everything that should become an HTTP error response.

Services (which know nothing about HTTP) raise ServiceError; main.py turns
it into the same {"detail": ...} body FastAPI's HTTPException produces."""


class ServiceError(Exception):
    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail
