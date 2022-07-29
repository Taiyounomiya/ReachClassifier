""" Utility functions for EDA_SampleReaches3 notebook. Written by Emily Nguyen, Brett Nelson, Nicholas Chin,
    Lawrence Berkeley national labs / U C Berkeley."""
# Filters dataframe for relevant time-information.
# Returns if the Trial, Rat, Session, Date in dataframe is found.
# NOTE: Rat & Session are strings, use "" when filtering.
# NOTE: For matching pairs, please use analyzedf.
# Example: # filterdf(0, "'"RM16", "S1", 17)
def filterdf(df, trial, rat, session, date):
    rr = df.loc[df['Date'] == date]
    rr1 = rr.loc[rr['Session'] == session]
    new_df = rr1.loc[rr1['Rat'] == rat]
    dff = new_df.loc[new_df['Trial']==trial]
    if dff.shape[0]==0:
        print(f"NO matching Trial was found for {trial, rat, session, date}")
    return dff

#FILTER
#filteredData = df.drop(df.columns.difference(selected_col_names), axis=1)
#filteredData.head()

# SELECT ALL
#Save Plots as PDF
def plt_to_pdf(figures, filename):
    # save multiple figures in one pdf file
    with matplotlib.backends.backend_pdf.PdfPages(filename) as pdf:
        for fig in figures:
            print(fig)
            pdf.savefig(fig)