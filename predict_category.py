import pickle

# Load model
with open("product_category_model.pkl", "rb") as f:
    model = pickle.load(f)

while True:
    title = input("Unesi naziv proizvoda: ")
    
    if title.lower() == "exit":
        break
    
    prediction = model.predict([title])[0]
    print(">>> Predviđena kategorija:", prediction)