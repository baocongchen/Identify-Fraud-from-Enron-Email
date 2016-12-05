#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """

    ### your code goes here
    point_count = len(predictions)
    ten_percent = int(round(point_count / 10.0, 0))
    errors = ((predictions - net_worths)**2).tolist()
    ages = ages.tolist()
    net_worths = net_worths.tolist()
    predictions = predictions.tolist()
    for i in range(ten_percent):
        ind = errors.index(max(errors))
        del(errors[ind]); del(predictions[ind]); del(ages[ind]); del(net_worths[ind])
    cleaned_data = zip(ages, net_worths, errors)
    return cleaned_data

