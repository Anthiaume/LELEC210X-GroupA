from load_data import load_data

locals = ["vinikot"]
speakers = ["JBL Flip 5 - Auguste - spec_20_20"] 
df = load_data(locals, speakers)

print(df.head())