"""OpenAlpr Model"""

alpr = None

def GetLP(image):
    global alpr
    if alpr is None:
        import sys
        from openalpr import Alpr
        alpr = Alpr("us", "/usr/share/openalpr/config/openalpr.defaults.conf", "/usr/share/openalpr/runtime_data")
        if not alpr.is_loaded():
            print('Error loading OpenALPR')
            sys.exit(1)
        alpr.set_top_n(20)

    if image is None:
        return None
    results = alpr.recognize_ndarray(image)['results']
    if len(results) == 0:
        return None
    if results[0]['confidence'] < 75:
        return None
    return results[0]['plate']