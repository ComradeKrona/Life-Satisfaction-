import pandas as pd
from pip._vendor.distlib.compat import raw_input

def readWEOFile():
    print("Reading \"WEO_Data.csv\"")

    # Read the File into the DataFrame
    worldData = pd.read_csv("WEO_Data.csv", sep="	", encoding="ISO-8859-1")
    # Dropping the Columns/Row that are useless
    worldData = worldData.drop(
        ['Subject Descriptor', 'Units', 'Scale', 'Country/Series-specific Notes', 'Estimates Start After'], axis=1)
    worldData = worldData.drop(189)
    # Renaming the columns
    worldData = worldData.rename(
        columns={"Country": "Country (Full Name)", "2015": "Gross Domestic Product per Capita (USD)"})

    print(worldData.head(), "\n", "\n", worldData.tail(), "\n")

    return worldData

def importDataFiles():
    # Create the main DataFrame
    dataFrame = readWEOFile()
    print("Reading \"Life metrics.csv\"")

    # Reads the file in the DataFrame
    lifeSatisfaction = pd.read_csv("Life metrics.csv")

    # Dropping the Columns/Row that are useless
    lifeSatisfaction = lifeSatisfaction.drop(
        ['LOCATION', 'INDICATOR', 'Measure', 'MEASURE', 'INEQUALITY', 'Unit', 'PowerCode Code', 'PowerCode Code',
         'PowerCode', 'Reference Period Code', 'Reference Period', 'Flag Codes', 'Flags'], axis=1)

    print(lifeSatisfaction.head(), "\n")
    print(list(lifeSatisfaction.columns), "\n")

    print("Starting to Combine the two DataFrames")
    for ind in lifeSatisfaction.index:
        # The Column where the value will be stored
        header = str(lifeSatisfaction['Indicator'][ind]) + " - " + str(
            lifeSatisfaction['Inequality'][ind]) + " (" + str(lifeSatisfaction['Unit Code'][ind]) + ")"
        # The Value
        value = lifeSatisfaction['Value'][ind]

        # Try to find the index of the country's name in the main DataFrame
        # If TypeError, create a new row in the main DataFrame
        try:
            rowNumber = int(dataFrame.loc[dataFrame.isin([lifeSatisfaction['Country'][ind]]).any(axis=1)].index.values)
        except TypeError:
            print("Adding new country", "\n")
            dataFrame = dataFrame.append({'Country (Full Name)': lifeSatisfaction['Country'][ind]}, ignore_index=True)

            # Again, tries to find the index of the country's name in the main DataFrame
            rowNumber = int(dataFrame.loc[dataFrame.isin([lifeSatisfaction['Country'][ind]]).any(axis=1)].index.values)

        # If the column is not in the main DataFrame, add it and set all of its values equal to "Nan"
        if header not in list(dataFrame.columns):
            dataFrame.insert(len(list(dataFrame.columns)), str(header), "Nan")

        # Add the value to the main DataFrame at column name and row number
        dataFrame[header][rowNumber] = value

    # Dropping the Columns/Row that are useless
    dataFrame = dataFrame.drop(189)

    # Have to make sure that all of the GDP are stored as numbers, so that they can be plotted
    for x in range(0, len(dataFrame.index)):
        for y in range(1, len(dataFrame.columns)):
            dataFrame.iat[x, y] = float(str(dataFrame.iat[x, y]).replace(",", ""))

    # Writes the single DataFrame into a file for easier acess
    dataFrame.to_csv("combinedData.csv", index=False)

    return dataFrame

def readCombinedCSVFile():
    # Reads the combinedDataFile for easier access
    return pd.read_csv("combinedData.csv")

def cleanUpCombinedData():
    dataFrame = readCombinedCSVFile()

    removeList = []

    # Check each row to see if more than half of its columns are filled with "NaN" or blank values
    # If so, the row is add to the removeList (latered to be dropped from the dataFrame)
    for rowIndex in dataFrame.index:
        totalNan = 0

        for columnIndex in range(0, len(list(dataFrame.columns))):
            if str(dataFrame.iat[rowIndex, columnIndex]).lower() == "nan" or str(
                    dataFrame.iat[rowIndex, columnIndex]).lower() == "":
                totalNan += 1

        if (totalNan > (.5 * len(list(dataFrame.columns)))):
            removeList.insert(0, rowIndex)

    for item in removeList:
        dataFrame = dataFrame.drop(item)

    # Making sure the removeList is now completely empty
    removeList = []

    # Detects for outliers in the GDP column using the IQR
    # If there is an outlier, the program will drop that row
    GDPIQR = dataFrame["Gross Domestic Product per Capita (USD)"].quantile(.75) - dataFrame[
        "Gross Domestic Product per Capita (USD)"].quantile(.25)
    GDPmedian = dataFrame["Gross Domestic Product per Capita (USD)"].median()

    for index in dataFrame.index:
        GDP = float(dataFrame.at[index, "Gross Domestic Product per Capita (USD)"])

        if GDP > (GDPmedian + GDPIQR) or GDP < (GDPmedian - GDPIQR):
            removeList.insert(0, index)

    for item in removeList:
        dataFrame = dataFrame.drop([item])

    # Again, making sure the removeList is now completely empty
    removeList = []

    # Detects for outliers in the LS (ACSCORE) column using the IQR
    # If there is an outlier, the program will drop that row
    LSIQR = dataFrame["Life satisfaction - Total (AVSCORE)"].quantile(.75) - dataFrame[
        "Life satisfaction - Total (AVSCORE)"].quantile(.25)
    LSmedian = dataFrame["Life satisfaction - Total (AVSCORE)"].median()

    for index in dataFrame.index:
        LS = float(dataFrame.at[index, "Life satisfaction - Total (AVSCORE)"])

        if LS > (LSmedian + LSIQR) or LS < (LSmedian - LSIQR):
            removeList.insert(0, index)

    for item in removeList:
        dataFrame = dataFrame.drop([item])

    # Saves the data to a CSV file for easier access
    dataFrame.to_csv("filteredData.csv", index=False)

    return dataFrame

def readCleanedCombinedCSVFile():
    # Reads the filteredData file for easier access
    return pd.read_csv("filteredData.csv")

def singleVaribleModel(dataFrame):
    #Prepares the single variable dataFrame
    removeList = []

    #Drops all columns expect for name, GDP and Life Satisfaction total
    for index in range(0, len(list(dataFrame.columns))):
        if str(list(dataFrame.columns)[index]) != "Life satisfaction - Total (AVSCORE)" and str(list(dataFrame.columns)[
            index]) != "Gross Domestic Product per Capita (USD)" and str(list(dataFrame.columns)[
            index]) != "Country (Full Name)":
            removeList.append(list(dataFrame.columns)[index])

    for item in removeList:
        dataFrame = dataFrame.drop(item, axis=1)

    return dataFrame

from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
def linearRegression(dataFrame, xList=None):
    #Results late to give the linear regression model
    resultValues = list(dataFrame["Life satisfaction - Total (AVSCORE)"])

    #Drops the results values and the name
    temptDataFrame = dataFrame.drop(["Country (Full Name)", "Life satisfaction - Total (AVSCORE)"], axis=1)

    #number of rows in the given data Dataframe
    numberOfRows = len(list((dataFrame.index)))

    #Adds all the items in xList to the masterDataFrame
    for item in xList:
        temptDataFrame = temptDataFrame.append({'Gross Domestic Product per Capita (USD)': item}, ignore_index=True)

    #number of rows in the new combined dataFrame
    numberOfValueRows = len(list((temptDataFrame.index)))

    #All of the "inputs"
    GDPValues = list(dataFrame["Gross Domestic Product per Capita (USD)"])
    for item in xList:
            GDPValues.append(float(item))

    #Adds the min and max input to graph the linear regression line later
    temptDataFrame = temptDataFrame.append({'Gross Domestic Product per Capita (USD)': max(GDPValues)}, ignore_index=True)
    temptDataFrame = temptDataFrame.append({'Gross Domestic Product per Capita (USD)': min(GDPValues)}, ignore_index=True)

    #Total number of items in the final dataFrame
    temptColumnList = list(temptDataFrame.columns)

    #Constructs a pipeline to replace all the blank values with the median in the column and to scale some of the value down
    num_pipeline = Pipeline([('imputer', SimpleImputer(strategy="median")), ('std_scaler', StandardScaler())])
    full_pipeline = ColumnTransformer([("num", num_pipeline, temptColumnList)])

    #Applies the pipeline to all of the items in dataFrame
    transformedPipeline = pd.DataFrame(data=full_pipeline.fit_transform(temptDataFrame), columns=temptColumnList)

    #Seperate the dataFrame back into its three parts
    lineDataFrame = transformedPipeline.iloc[numberOfValueRows:]
    xDataFrame = transformedPipeline.iloc[numberOfRows:numberOfValueRows]
    temptDataFrame = transformedPipeline.iloc[0:numberOfRows]

    #Forms the linear regression model
    linearRegression = LinearRegression()
    linearRegression.fit(X=temptDataFrame, y=resultValues)

    ylist = []

    #Stores the model's prediction for the xlist inputs
    for index in range(0, len(list(xDataFrame.index))):
        ylist.append(linearRegression.predict(xDataFrame.iloc[[index]]))

    #Gets the model's prediction to the graph the line
    xLineList = [max(GDPValues), min(GDPValues)]
    yLineList = []

    for index in range(0, len(list(lineDataFrame.index))):
        yLineList.append(linearRegression.predict(lineDataFrame.iloc[[index]]))

    graphingTheData(dataFrame, xList, ylist, xLineList, yLineList)

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
def graphingTheData(dataframe, xList=None, yList=None, xLineList=None, yLineList=None):
    #Formatting for Graph
    font = {'family': 'DejaVu Sans','size': 8}
    plt.rc('font', **font)

    #Shift the texts so that you can see it better
    YSHIFT = .03
    XSHIFT = 150

    #Creating the figure
    fig, ax = plt.subplots()

    #Adding the given scatter plot
    ax.scatter(dataframe["Gross Domestic Product per Capita (USD)"], dataframe["Life satisfaction - Total (AVSCORE)"], c='Blue')

    #Adding the names of the countries to the graph
    for index in range(0, len(dataframe.index)):
        plt.annotate(dataframe.at[list(dataframe.index)[index], 'Country (Full Name)'], (
            dataframe.at[list(dataframe.index)[index], 'Gross Domestic Product per Capita (USD)'] + XSHIFT,
            dataframe.at[list(dataframe.index)[index], 'Life satisfaction - Total (AVSCORE)'] + YSHIFT))

    #Adding the regression linear
    line = mlines.Line2D(xdata=xLineList, ydata=yLineList, color='Green')
    ax.add_line(line)

    #Adding the predicted values
    ax.scatter(xList, yList, c="Red")

    #Added the coordinates for the predicted values
    for index in range(0, len(xList)):
        plt.annotate(("(" + str(xList[index]) +", " + str(yList[index]) +")"), (xList[index] + XSHIFT, yList[index] + YSHIFT))

    #Labeling and the titling the graph
    plt.title('Comparing a Country\'s Life Satisfaction to their Gross Domestic Product per Capita')
    plt.xlabel('Gross Domestic Product per Capita (USD)')
    plt.ylabel('Life satisfaction - Total (AVSCORE)')

    plt.show()

def main():
    print(
        "Input can only take integers and floats. You can add multiple numbers at a time by seperating them with commas")
    print("Example: \"50000, 42000, 1300.0\"", "\n")
    print("(27000 (Cypriot) is already added) )")

    stopLoop = False
    xList = [27000]

    print("Input integers and floats to feed into the model. Type \"Predict\" or \"Run\" to continue")
    while not stopLoop:
        xListValue = raw_input("User Input: ").strip().lower()

        if xListValue == "predict" or xListValue == "run":
            stopLoop = True
        else:
            for variables in xListValue.split(","):
                try:
                    xList.append(float(variables.strip()))
                except ValueError:
                    print("\"" + str(variables.strip()) + "\" is not an acceptable input. Please try again.")
        print()

    print("Input List: " + str(xList), "\n")

    singleVariableModel = True

    print("Form a Linear Regression Model based on a single variable? (True/T or Flase/F)")
    print("(I would suggest single variable to start with)")
    while stopLoop:
        booleanValue = raw_input("User Input: ").strip().lower()

        if booleanValue == "true" or booleanValue == "t":
            singleVariableModel = True
            stopLoop = False
        elif booleanValue == "false" or booleanValue == "f":
            singleVariableModel = False
            stopLoop = False
        else:
            print("\"" + str(booleanValue) + "\" is not an acceptable input. Please try again.", "\n")

    if singleVariableModel:
        dataSet = singleVaribleModel(readCleanedCombinedCSVFile())
    else:
        dataSet = readCleanedCombinedCSVFile()

    linearRegression(dataSet, xList)

if __name__ == "__main__":
    main()
