## Trash Info

This is used to associate the categories that we will be using for SSD with results from the Sacramento Waste Wizard.

searchList is a dict with sublists of objects to search in the Sacramento Waste Wizard.

searchList = {
    'garbage': [
        "garbage bag"]
    ],
    'recycling': [
        'Clear glass']
}

## get_id(categoryStr)
We first search the fuzzy search waste wizard by swapping our category with suggest=paper

    https://api.recollect.net/api/areas/Sacramento/services/waste/pages?suggest=paper&type=material&set=default&include_links=true&locale=en-US&accept_list=true&_=1666888380289

This returns a list of suggestions, right now we just take the first id (eg 473087). There is some room for improvement to ensure that the correct category is selected.

## searchForBin(id)
Then the id is passed to this query which will return a larger JSON with lots of trash metadata. Sorting is done to find the single string value of what to do. There is lots of other data that needs to be parsed and returned in a more meaningful pattern. Or we could actually return the waste wizard location.

## searchForDescription(id)
TODO: Returns the description for an item
https://api.recollect.net/api/areas/Sacramento/services/waste/pages/en-US/473087.json

## recollect_api(url)
API calls code reuse

## Possible future improvements
- make search return a list of suggestions
- Return built html for each object so it looks like the waste wizard