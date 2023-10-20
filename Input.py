import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

def load_data():
    df=pd.read_csv('2008.csv')
    deduplicate_df = df.drop_duplicates(keep='first')
    deduplicate_df['WeatherDelay'] = deduplicate_df['WeatherDelay'].fillna(0)
    deduplicate_df['CarrierDelay'] = deduplicate_df['CarrierDelay'].fillna(0)
    deduplicate_df['NASDelay'] = deduplicate_df['NASDelay'].fillna(0)
    deduplicate_df['LateAircraftDelay']=deduplicate_df['LateAircraftDelay'].fillna(0)
    deduplicate_df['SecurityDelay'] = deduplicate_df['SecurityDelay'].fillna(0)
    deduplicate_df['ArrDelay'] = deduplicate_df['ArrDelay'].fillna(0)
    Cancelled_df = deduplicate_df.loc[deduplicate_df['Cancelled']==1]
    Non_Cancelled_df = deduplicate_df.loc[deduplicate_df['Cancelled']!=1]
    Non_Cancelled_df = Non_Cancelled_df.drop(columns = ['Cancelled','CancellationCode'])
    Non_Diverted_df = Non_Cancelled_df.loc[Non_Cancelled_df['Diverted']!=1]
    Non_Diverted_df = Non_Diverted_df.drop(columns = ['Diverted'])
    Non_Diverted_df['TotalDelay'] = Non_Diverted_df['WeatherDelay']+Non_Diverted_df['CarrierDelay']+Non_Diverted_df['NASDelay']+Non_Diverted_df['LateAircraftDelay']+Non_Diverted_df['SecurityDelay']
    Non_Diverted_df = Non_Diverted_df.drop(columns = ['WeatherDelay','CarrierDelay','NASDelay','LateAircraftDelay','SecurityDelay'])
    Cleaned_df = Non_Diverted_df.drop(columns=['DayOfWeek'])
    Cleaned_df = Cleaned_df.drop(columns = ['Year','Month','DayofMonth','UniqueCarrier','TailNum','Origin','Dest'])
    return Cleaned_df

Cleaned_df=load_data()

def show_explore_page():
    st.title("Flight Delay Analysis")
    st.write("## EDA of flight delay")
    st.write("### Glimpse of Data")
    st.dataframe(Cleaned_df.head(50))
    st.write("### Visualizations!")
    
    st.write("### Frequency Plots")
    figure1 = plt.figure(figsize =(4,4))
    Cleaned_df['FlightNum'].value_counts()[0:10].plot(kind='bar')
    plt.xlabel('Flight Number')
    plt.ylabel('Frequency')
    plt.xticks(rotation = 90)
    plt.title('Frequency of top 10 flights')
    st.pyplot(figure1)
    st.write ("#### We can infer from the above graph that Flight number 511 has most number of entries in our datatset followed by 16 and so on.")
    
    figure2 = plt.figure(figsize =(4,4))
    Cleaned_df.groupby('FlightNum')['TotalDelay'].sum().nlargest(10).plot(kind='bar')
    plt.xlabel('Flight Number')
    plt.ylabel('Delay in mins')
    plt.xticks(rotation = 90)
    plt.title('Top 10 Delayed flights')
    st.pyplot(figure2)
    st.write ("#### We can infer from the above graph that Flight Number 378 has been delayed the most with a total delay of 30000 mins in our dataset.")
 
    figure3 = plt.figure(figsize =(4,4))
    plt.hist(Cleaned_df['TotalDelay'],bins=20)
    plt.xlabel('Total Delay in mins')
    plt.ylabel('Distribution of delay across the dataset')
    plt.xticks(rotation = 90)
    plt.title('Distribution of Delay across the Dataset')
    st.pyplot(figure3)
    st.write("#### We can infer from the histogram that most flight delays are between 0 mins to 2500 mins")


    st.write("## Predictions")
    Y = Cleaned_df["TotalDelay"]
    #X = Cleaned_df.loc[:,Cleaned_df.columns!="TotalDelay"]
    X = Cleaned_df[['DepTime','CRSDepTime','ArrTime','FlightNum','AirTime','Distance']]
    std_scale = StandardScaler()
    X=std_scale.fit_transform(X)
    X_Train,X_Test,Y_Train, Y_Test = train_test_split(X,Y,test_size=0.2)
    forest_reg = RandomForestRegressor(n_estimators=10,random_state=0)
    forest_reg.fit(X_Train,Y_Train)
    Y_Pred_forest = forest_reg.predict(X_Test)
    forest_reg.score(X_Test,Y_Test)

    Dept_Time = st.number_input("Input Departure time (hhmm):")
    CRS_DepTime = st.number_input("Input Actual Departure time (hhmm):")
    Arr_Time = st.number_input("Input Arrival time (hhmm):")
    Flight_Num = st.number_input("Input Flight Number:")
    Air_Time = st.number_input("Input Air time (hhmm):")
    Dist = st.number_input("Input Distance in miles:")
    cal = st.button("Calculate Delay")
    if cal:
        input_list =  [Dept_Time, CRS_DepTime, Arr_Time,Flight_Num,Air_Time,Dist]
        input_np = np.array(input_list)
        input_np = input_np.astype(float).reshape(1,6)
        y_new = forest_reg.predict(input_np)
        y_new[0]
        if y_new[0]<0:
            y_new[0] = 0
        st.subheader(f'Estimated flight delay is mins {y_new[0]:.2f}')