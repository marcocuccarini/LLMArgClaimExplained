import json

# Open and read the JSON file
with open("predictions_K5_gpt-oss-20b.json", "r") as file:
    data = json.load(file)  # Converts JSON into Python dict/list


correct=0
allof=0
notproduced=0


for i in data:

    print(i)

    if not ( i["prediction"] == "NO_EVIDENCE" ):

        if i["prediction"] == i["ground_truth"]:

            correct +=1

        allof +=1

    else:

        notproduced += 1

print(correct/allof)
print(notproduced)


