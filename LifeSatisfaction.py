import pandas as pd
import matplotlib.pyplot as plt
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

# Reads the combinedDataFile for easier access
def readCombinedCSVFile():
    return pd.read_csv("combinedData.csv")


from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
def cleanUpCombinedData():
    dataFrame = readCombinedCSVFile()

    removeList = []

    #Check each row to see if more than half of its columns are filled with "NaN" or blank values
    #If so, the row is add to the removeList (latered to be dropped from the dataFrame)
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

    #Making sure the removeList is now completely empty
    removeList = []

    #Detects for outliers in the GDP column using the IQR
    #If there is an outlier, the program will drop that row
    GDPIQR = dataFrame["Gross Domestic Product per Capita (USD)"].quantile(.75) - dataFrame[
        "Gross Domestic Product per Capita (USD)"].quantile(.25)
    GDPmedian = dataFrame["Gross Domestic Product per Capita (USD)"].median()

    for index in dataFrame.index:
        GDP = float(dataFrame.at[index, "Gross Domestic Product per Capita (USD)"])

        if GDP > (GDPmedian + GDPIQR) or GDP < (GDPmedian - GDPIQR):
            removeList.insert(0, index)

    for item in removeList:
        dataFrame = dataFrame.drop([item])

    #Again, making sure the removeList is now completely empty
    removeList = []

    #Detects for outliers in the LS (ACSCORE) column using the IQR
    #If there is an outlier, the program will drop that row
    LSIQR = dataFrame["Life satisfaction - Total (AVSCORE)"].quantile(.75) - dataFrame[
        "Life satisfaction - Total (AVSCORE)"].quantile(.25)
    LSmedian = dataFrame["Life satisfaction - Total (AVSCORE)"].median()

    for index in dataFrame.index:
        LS = float(dataFrame.at[index, "Life satisfaction - Total (AVSCORE)"])

        if LS > (LSmedian + LSIQR) or LS < (LSmedian - LSIQR):
            removeList.insert(0, index)

    for item in removeList:
        dataFrame = dataFrame.drop([item])

    #Saves the data to a CSV file for easier access
    dataFrame.to_csv("filteredData.csv", index=False)

    return dataFrame

def readCleanedCombinedCSVFile():
    return pd.read_csv("filteredData.csv")


def pipeline(dataFrame):
    columnList = list(dataFrame.columns)
    columnList.remove("Country (Full Name)")

    num_pipeline = Pipeline([('imputer', SimpleImputer(strategy="median"))
                             #   , ('std_scaler', StandardScaler())
                             ])
    full_pipeline = ColumnTransformer([("num", num_pipeline, columnList)])
    transformedPipeline = full_pipeline.fit_transform(dataFrame)
    # transformedPipeline.to_csv("fittedData.csv", index=False)

    return transformedPipeline


def graphingTheData(dataframe, xList=None, yList=None, line=None):
    worldData = dataframe

    # Add given Data Points
    worldData.plot.scatter(x='Gross Domestic Product per Capita (USD)', y='Life satisfaction - Total (AVSCORE)',
                           color='DarkBlue')

    # Add linearRegressionLine

    # Add predicted values
    # for index in range(0, len(xList)):

    for index in range(0, len(worldData.index)):
        plt.annotate(worldData.at[list(worldData.index)[index], 'Country (Full Name)'], (
            worldData.at[list(worldData.index)[index], 'Gross Domestic Product per Capita (USD)'],
            worldData.at[list(worldData.index)[index], 'Life satisfaction - Total (AVSCORE)']))

    GDPValues = max(list(worldData["Gross Domestic Product per Capita (USD)"]))

    plt.plot(list(worldData["Gross Domestic Product per Capita (USD)"]),
             list(worldData["Life satisfaction - Total (AVSCORE)"]), color="DarkRed")

    plt.title('Life Satisfaction V GDP')
    plt.xlabel('GDP per Captia')
    plt.ylabel('Life Satisfaction - Total (AVSCORE)')

    plt.show()

    # What I Still Need to Do
    # User input function


from sklearn.linear_model import LinearRegression


def linearRegression(dataFrame, GDPValues, xList=None):
    print(GDPValues)

    columnList = list(dataFrame.columns)
    columnList.remove("Country (Full Name)")

    num_pipeline = Pipeline([('imputer', SimpleImputer(strategy="median"))
                                , ('std_scaler', StandardScaler())
                             ])
    full_pipeline = ColumnTransformer([("num", num_pipeline, columnList)])
    transformedPipeline = full_pipeline.fit_transform(dataFrame)

    lsValues = list(dataFrame["Life satisfaction - Total (AVSCORE)"])

    rowBefore = len(list(dataFrame.index))

    Cypriot = 27000

    dataFrame = dataFrame.append({'Gross Domestic Product per Capita (USD)': Cypriot}, ignore_index=True)

    rowAfter = len(list(dataFrame.index))

    print(dataFrame.tail())

    print("LSValues: ", lsValues)

    linearRegression = LinearRegression()
    linearRegression.fit(transformedPipeline, lsValues)

    for index in range(rowBefore, rowAfter):
        prediction = linearRegression.predict(
            full_pipeline.transform(
                dataFrame.loc[dataFrame['Gross Domestic Product per Capita (USD)'] == Cypriot]
            ))
        print("Predictions:", str(prediction))

    print(linearRegression.coef_)
    print(linearRegression.intercept_)

    return prediction, linearRegression


def arrayToDataFrame(array, columnList, names):
    dataFrameFilledIn = pd.DataFrame(data=array, columns=columnList)
    dataFrameFilledIn.insert(0, "Country (Full Name)", names)

    return dataFrameFilledIn

def main():
    print("Input can only take integers and floats. You can add multiple numbers at a time by seperating them with commas")
    print("Example: \"50000, 42000, 1300.0\"", "\n")

    stopLoop = False
    xList = []

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

    print("Form a Linear Regression Model based on a single varaible? (True/T or Flase/F)")
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

    dataSet = readCleanedCombinedCSVFile()
    columnList = list(dataSet.columns)
    names = dataSet["Country (Full Name)"].tolist()
    GDPValues = list(dataSet["Gross Domestic Product per Capita (USD)"])

    graphingTheData(dataSet)

    linearRegression(dataSet, GDPValues, xList)

    singleOrMultiple = input("Single or Multiple Linear Regressoin (Single or Multiple): ")

    if singleOrMultiple.lower() == "single":
        print("There")
    else:
        print("Hello")

if __name__ == "__main__":
    main()
