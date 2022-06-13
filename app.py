import streamlit as st
from datetime import date

import yfinance as yf
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go

START="2015-01-01"
TODAY=date.today().strftime("%Y-%m-%d")

st.title("stock prediction")
stocks=("SBIN.NS","AAPL","GOOG","MSFT","GME")
selected_stocks=st.selectbox("select dataset for prediction",stocks)

n_years=st.slider("years of prediction:",1,4)
period=n_years*365

def load_data(ticker):
    data=yf.download(ticker,START,TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state=st.text("load data")
data=load_data(selected_stocks)
data_load_state.text("load data ...done!")

st.subheader('Raw dta')
st.write(data.tail())


def plot_raw_data():
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'],y=data['Open'],name='stock_open'))
    fig.add_trace(go.Scatter(x=data['Date'],y=data['Close'],name='stock_open'))
    fig.layout.update(title_text="time series data",xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()


#forcasting

df_train=data[['Date','Close']]
df_train=df_train.rename(columns={"Date":"ds","Close":"y"})
m=Prophet()
m.fit(df_train)
future=m.make_future_dataframe(periods=period)
forecast=m.predict(future)


st.subheader('Forecast data')
st.write(forecast.tail())

st.write("forecastdata")
fig1=plot_plotly(m,forecast)
fig1.add_trace(go.Scatter(x=data['Date'],y=data['Open'],name='stock_open'))
fig1.add_trace(go.Scatter(x=data['Date'],y=data['Close'],name='stock_open'))
st.plotly_chart(fig1)

st.write("forecast component")

fig2=m.plot_components(forecast)
st.write(fig2)