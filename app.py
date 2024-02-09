import streamlit as st
import pandas as pd
import dt_prcss

def main():
    st.title("Large and Sparse matrix Storasge and retrival")

    # File uploader
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    dummy_data = st.checkbox("Use dummy data instead")
    if dummy_data:
        uploaded_file = "dummy_data_1.csv"

    if uploaded_file is not None:


        # Read the CSV file
      df = pd.read_csv(uploaded_file)
      test = df.isin([0.0,1.0]).all()
      if test.any() ==False:
        st.write("upload csv file in coorect format. CSV file should contain M x N matrix of 0 or 1")
      else:
        ext = df.shape[1] % 8
        if ext != 0 :
            for i in range(8-ext):
                df['c'+str(i)] = 0
        dt = dt_prcss.dt(df)

        num_indx = st.number_input("Type the index of the supplier to know service availability", min_value=1, step=1, value=5)

        row = dt.get_indx(num_indx)
        pincode = st.number_input("Type the index of the pincode to know service availability", min_value=1, step=1, value=5)
        # Display the first row of the CSV file
        st.write("Servicability of supplier for the pincode:") 
        if row[pincode] == 1:
            st.write("Service available")
        else:
            st.write("Service is not available")
        #st.write(row[:10])

if __name__ == "__main__":
    main()
