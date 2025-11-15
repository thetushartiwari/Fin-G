from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from .k_mean import predict_risk
from .st_perform import get_top_stocks 

def index(request):
    return render(request, 'index.html') 

def success_stories(request):
    return render(request, "success_stories.html")

def classify_investment(savings, risk_score):
    allocation = {
        "mutual_funds": 0,
        "stocks":0,
        "sips": 0,
        "Govt_bonds": 0
    }

    if risk_score in [1, 2, 3]:  
        allocation["mutual_funds"] = 0.5 * savings
        allocation["stocks"] = 0.2 * savings
        allocation["sips"] = 0.2 * savings
        allocation["Govt_bonds"] = 0.1 * savings
    elif risk_score in [4, 5, 6]:  
        allocation["mutual_funds"] = 0.3 * savings
        allocation["stocks"] = 0.4 * savings
        allocation["sips"] = 0.2 * savings
        allocation["Govt_bonds"] = 0.1 * savings
    else:  
        allocation["mutual_funds"] = 0.1 * savings
        allocation["stocks"] = 0.6 * savings
        allocation["sips"] = 0.2 * savings
        allocation["Govt_bonds"] = 0.1 * savings

    return allocation

@csrf_exempt
def analyze_user_investment(request):
    if request.method == "POST":
        try:
            income = float(request.POST.get("income"))
            savings = float(request.POST.get("savings"))
            expenses = float(request.POST.get("expenses"))
            age = int(request.POST.get("age"))
            interest = request.POST.get("stockPreference")

            if None in [income, savings, expenses, age, interest]:
                return render(request, "index.html", {"error": "⚠ Please fill in all fields."})


            user_input = [float(income), float(savings), float(expenses), int(age)]
            risk_score = predict_risk(user_input)
            allocation = classify_investment(float(savings), risk_score)
            top_stocks = get_top_stocks(interest, risk_score)

            return render(request, "index.html", {
            "risk_score": risk_score,
            "allocation": allocation,
            "top_stocks": top_stocks
        })
        

        except Exception as e:
            return render(request, "index.html", {"error": str(e)})

    return render(request, "index.html")


