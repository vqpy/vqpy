
def within_regions(regions):
    def point_within_regions(coordinate):
        from shapely.geometry import Point, Polygon
        point = Point(coordinate)
        for region in regions:
            poly = Polygon(region)
            if point.within(poly):
                return True
        return False
    return point_within_regions
