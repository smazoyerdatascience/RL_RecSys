from fastapi import FastAPI

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/get_lin_ucb")
async def get_linucb_reco():
    """Return LinUCB recommandations list to UI"""
    return {"message": "lin_ucb_reco"}

@app.get("/get_content_based")
async def get_contentbased_reco():
    """Return Content Based recommandations list to UI"""
    return {"message": "content_based_reco"}