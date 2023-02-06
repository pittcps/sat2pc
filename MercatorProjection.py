from __future__ import division
import math
import data_util
MERCATOR_RANGE = 256

def  bound(value, opt_min, opt_max):
  if (opt_min != None): 
    value = max(value, opt_min)
  if (opt_max != None): 
    value = min(value, opt_max)
  return value

def  degreesToRadians(deg) :
  return deg * (math.pi / 180)

def  radiansToDegrees(rad) :
  return rad / (math.pi / 180)

class G_Point :
    def __init__(self,x=0, y=0):
        self.x = x
        self.y = y

class G_LatLng :
    def __init__(self,lt, ln):
        self.lat = lt
        self.lng = ln

class MercatorProjection :
    def __init__(self) :
      self.pixelOrigin_ =  G_Point( MERCATOR_RANGE / 2, MERCATOR_RANGE / 2)
      self.pixelsPerLonDegree_ = MERCATOR_RANGE / 360
      self.pixelsPerLonRadian_ = MERCATOR_RANGE / (2 * math.pi)

    def fromLatLngToPoint(self, latLng, opt_point=None) :
      point = opt_point if opt_point is not None else G_Point(0,0)
      origin = self.pixelOrigin_
      point.x = origin.x + latLng.lng * self.pixelsPerLonDegree_
      # NOTE(appleton): Truncating to 0.9999 effectively limits latitude to
      # 89.189.  This is about a third of a tile past the edge of the world tile.
      siny = bound(math.sin(degreesToRadians(latLng.lat)), -0.9999, 0.9999)
      point.y = origin.y + 0.5 * math.log((1 + siny) / (1 - siny)) * -self.pixelsPerLonRadian_
      return point

    def fromPointToLatLng(self,point) :
      origin = self.pixelOrigin_
      lng = (point.x - origin.x) / self.pixelsPerLonDegree_
      latRadians = (point.y - origin.y) / -self.pixelsPerLonRadian_
      lat = radiansToDegrees(2 * math.atan(math.exp(latRadians)) - math.pi / 2)
      return G_LatLng(lat, lng)

def get_pixel_xy_from_lat_lon(center_latlon, point_global_xy, zoom, mapWidth, mapHeight, srs, offset, lidar_scale, alignment):
    scale = 2**zoom
    proj = MercatorProjection()

    corners = getCorners(center_latlon, zoom, mapWidth, mapHeight)
    NW_lat = corners['N']
    NW_lon = corners['W']
    plat, plon = data_util.get_lat_lon_from_xy((point_global_xy.x + offset[0] - alignment[0]) * lidar_scale[0], (point_global_xy.y + offset[1]+ alignment[1]) * lidar_scale[1] , srs)
    p = proj.fromLatLngToPoint(G_LatLng(plat, plon))
    nw = proj.fromLatLngToPoint(G_LatLng(NW_lat, NW_lon))
    return (p.x-nw.x)*scale, (p.y-nw.y)*scale

#calculates lat lon of pixel using info of center point
def get_pixel_lat_lon(center, pixel_x, pixel_y, zoom, mapWidth, mapHeight):
    scale = 2**zoom
    proj = MercatorProjection()
    centerPx = proj.fromLatLngToPoint(center)
    NewPoint = G_Point(centerPx.x- (mapWidth/2 - pixel_x)/scale, centerPx.y - (mapHeight/2 - pixel_y)/scale)
    
    NewPoint = proj.fromPointToLatLng(NewPoint)

    return NewPoint.lat, NewPoint.lng

def getCorners(center, zoom, mapWidth, mapHeight):
    scale = 2**zoom
    proj = MercatorProjection()
    centerPx = proj.fromLatLngToPoint(center)
    SWPoint = G_Point(centerPx.x-(mapWidth/2)/scale, centerPx.y+(mapHeight/2)/scale)
    SWLatLon = proj.fromPointToLatLng(SWPoint)
    NEPoint = G_Point(centerPx.x+(mapWidth/2)/scale, centerPx.y-(mapHeight/2)/scale)
    NELatLon = proj.fromPointToLatLng(NEPoint)
    return {
        'N' : NELatLon.lat,
        'E' : NELatLon.lng,
        'S' : SWLatLon.lat,
        'W' : SWLatLon.lng,
    }