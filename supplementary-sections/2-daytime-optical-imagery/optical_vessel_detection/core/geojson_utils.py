

def detections_to_geojson(path):
    out_path = path.replace('.json', '.geojson')
    features = []
    obj = {
"type": "FeatureCollection",
"crs": { "type": "name", "properties": { "name": "urn:ogc:def:crs:OGC:1.3:CRS84" } },
"features": features }
    def add_feature(scene_id, lon1, lat1, lon2, lat2):
        features.append({  "type": "Feature", 
                          "properties": { "type": 1, "scene_id": scene_id, "feat_type": "is_boat" }, 
                          "geometry": { "type": "LineString", 
                          "coordinates": [ 
                          [lon1, lat1], [lon2, lat2 ] ] } })
    with open(path) as f:
        detections = json.loads(f.read())
    scene_id = detections['scene']
    for x in detections['detections']:
        lon, lat, length, angle = x
        dx = np.cos(angle) * length
        dy = np.sin(angle) * length
        to_degrees_lat = 1.0 / 1852. / 60. 
        to_degrees_lon = to_degrees_lat / np.cos(np.radians(lat))
        dlon = 0.5 * dx * to_degrees_lat
        dlat = 0.5 * dy * to_degrees_lon
        add_feature(scene_id, lon - dlon, lat - dlat, lon + dlon, lat + dlat)
    with open(out_path, 'w') as f:
        f.write(json.dumps(obj))