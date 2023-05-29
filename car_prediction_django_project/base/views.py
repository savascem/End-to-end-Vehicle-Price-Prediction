from django.shortcuts import render
import pickle
import pandas as pd
from base.pipeline.pipeline import pipeline
import plotly.graph_objs as go
import plotly.offline as opy

def home(request):
    return render(request, 'index.html')

def getPredictions(X, price, model_, year):
    
    # predict with model
    model_path = 'base/files/model.pkl'
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    model = model["model"]
    prediction = model.predict(X)

    # Is the estimated price really cheap or expensive?
    stt = ""
    if prediction[0] > price:
        stt = "cheap"
    else:
        stt = "expensive"

    # returns the final edited version of the dataframe
    df = pd.read_csv("base/files/website_df.csv")

    # Calculates vehicle price averages by make, model and year 
    # price averaging returns the vehicles closest to the vehicle entered by the user.
    similar = df.groupby(["Brand", "Model_1", "Year"]).agg({"Primary Price" : "mean"})
    similar.reset_index(inplace=True)
    analyze_df = similar.copy()
    similar["Primary Price"] = abs(similar["Primary Price"] - price)
    list_similar = similar.sort_values(by="Primary Price", ascending=True)[2:8][["Brand", "Model_1", "Year"]].values

    return int(prediction[0]), stt, list_similar, analyze_df, df


def result(request):
    url = str(request.GET['url'])

    # model pipeline
    X, brand, model_, price, year = pipeline(url)
    result, stt, list_similar, analyze_df, df = getPredictions(X, price, model_, year)

    # absolute value of the difference between the actual value and the estimate
    diff = abs(price-result)

    # price change of the model over the years
    prices = analyze_df[analyze_df["Model_1"] == model_]["Primary Price"].values
    years = analyze_df[analyze_df["Model_1"] == model_]["Year"].values
    data = go.Bar(x=years, y=prices)
    layout = go.Layout(
    title=f"Change in average prices of {brand} - {model_} vehicles according to years of production.",
    xaxis=dict(title="Year", tickmode="linear", tick0=2000, dtick=1),
    yaxis=dict(range=[0, max(prices) + 1000], title="Primary Price")
    )
    fig = go.Figure(data=data, layout=layout)
    plot_div = opy.plot(fig, auto_open=False, output_type='div')

    # Average price distribution of brand vehicles
    pie_df = df[(df["Brand"] == brand) & (df["Year"] == year)]
    pie_df = pie_df.groupby(["Model_1"]).agg({"Primary Price" : "mean"})
    pie_df.reset_index(inplace=True)
    prices = pie_df["Primary Price"].values
    brnd_ = pie_df["Model_1"].values

    sort_model = pie_df.sort_values(by="Primary Price", ascending=False)[["Model_1", "Primary Price"]]
    count = 0
    for ind, m in enumerate(sort_model["Model_1"]):
        if m == model_:
            count=ind+1

    highlight_model = model_
    highlight_index = list(brnd_).index(highlight_model)

    colors = ['rgba(255, 0, 0, 1)', 'rgba(0, 255, 0, 1)', 'rgba(0, 0, 255, 1)', 'rgba(128, 128, 128, 1)']
    line_colors = ['rgba(0, 0, 0, 1)'] * len(brnd_)
    line_colors[highlight_index] = 'rgba(255, 255, 255, 1)'

    data = go.Pie(
        labels=brnd_,
        values=prices,
        marker=dict(colors=colors, line=dict(color=line_colors, width=2))
    )

    layout = go.Layout(
        title=f"{ year } { model_ } is the { count }th vehicle with the highest average price among { brand } brand.",
        height=900,
        width=900
    )

    fig = go.Figure(data=data, layout=layout)
    plot_div_2 = opy.plot(fig, auto_open=False, output_type='div')

    return render(request, 'index.html', {'result': result, 
                                           'brand': brand, 
                                           'model': model_, 
                                           'price': price, 
                                           'stt': stt, 
                                           'diff': diff,
                                           'list_similar': list_similar,
                                           'plot_div': plot_div,
                                           'plot_div_2': plot_div_2,
                                           'count': count,
                                           'year': year,
                                           })

