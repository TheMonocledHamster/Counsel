from fastapi import FastAPI

app = FastAPI()

@app.get("/",methods=["POST"])
def root():
    pass