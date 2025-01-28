from fastapi import FastAPI

from src.api.routers import recommender

app = FastAPI(title="LLM Recommender API", version="1.0")

# Include API routers
app.include_router(recommender.router)


@app.get("/")
def root():
    return {"message": "Welcome to the Parenting Chatbot API!"}


# add helth check endpoint
@app.get("/health")
def health():
    return {"status": "healthy"}
