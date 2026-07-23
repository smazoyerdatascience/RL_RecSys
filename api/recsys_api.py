from fastapi import FastAPI

app = FastAPI()


@app.get("/")
async def main_page():
    return {"message": "Main page with recommandations and products selection"}

@app.get("/get_product_click")
async def get_product_click():
    """Modify context vector with the fact a product has been clicked on. 
    And return presentation page for the product"""
    return {"message": "product_click"}

@app.get("/get_product_buy")
async def get_product_buy():
    """Modify context vector with the fact a product has been bought. 
    And return to main page with 'thanks for purchase message'"""
    return {"message": "product_bought"}

@app.get("/get_lin_ucb")
async def get_lin_ucb():
    """Return LinUCB recommandations list to UI for test purposes"""
    return {"message": "lin_ucb_reco"}

@app.get("/get_content_based")
async def get_content_based():
    """Return Content Based recommandations list to UI for test purposes"""
    return {"message": "content_based_reco"}

