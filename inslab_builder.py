"""
Python module to construct an OpenQuake subduction in-slab source model using
the OpenQuake finite-rupture in volume methodolgy explained in

Weatherill, G., Pagani, M and Garcia, J. (2017) "Modelling In-slab Subduction
Earthquakes in PSHA: Current Practice and Challenges for the Future" in 
Proceedings of the 16th World Conference on Earthquake Engineering, 16WCEE,
Santiago, Chile, January 9th to 13th 2017
"""
import re
import numpy as np
from math import fabs, asin, atan, sqrt, pi
from linecache import getlines
from shapely import geometry
from shapely.geometry import LineString
from openquake.baselib.node import Node
from openquake.hazardlib import nrml
from openquake.hazardlib.geo import utils as geo_utils
from openquake.hazardlib.geo.geodetic import (azimuth,
                                              distance,
                                              geodetic_distance)
from openquake.hazardlib.pmf import PMF
from openquake.hazardlib.sourcewriter import (write_source_model,
                                              build_linestring_node)
from openquake.hazardlib.tom import PoissonTOM
from openquake.hazardlib.scalerel.strasser2010 import StrasserIntraslab
from openquake.hazardlib.geo import (RectangularMesh, Point, Line,
                                     ComplexFaultSurface, NodalPlane)
from openquake.hazardlib.mfd import EvenlyDiscretizedMFD, TruncatedGRMFD
from openquake.hazardlib.source.point import PointSource

def read_xyz_file(file_name):
    """
    Reads coordinates input in a simple long,lat,depth text file and returns
    as a numpy array
    """
    f = open(file_name,'r')
    lines = f.readlines()
    f.close()
    
    data = np.zeros((len(lines),3))
    for i,line in enumerate(lines):
        xyz = np.array(line.split(),dtype=float)
        data[i,:] = xyz

    return data


def read_upper_edge_file(filename):
    """
    Reads upper edge of a subduction interface set based on the Slab 1.0
    top edge file (usually named XXX_top.in) formatted as lon, lat, depth and
    return as an openquake.hazardlib.geo.Line object.

    Note longitudes are IDL centred in input file (i.e. from 0.0 to 360.0)
    """
    data = np.genfromtxt(filename)
    edge = []
    for row in data:
        if row[0] > 180.:
            row[0] = -(360. - row[0])
        edge.append(Point(row[0], row[1], fabs(row[2])))
    return Line(edge[::-1])


def read_contour_file(filename, min_depth=0., max_depth=120.):
    """
    Reads a set of edges as a Slab 1.0 contour file (usually named
    XXX_contours.in), formatted as lon, lat, depth, and return as a list of
    openquake Line objects

    :param float min_depth:
        Minimum depth (km) for consideration (edges shallower than this are
        ignored)

    :param float max_depth:
        Maximum depth for (km) consideration (edges deeper than this are
        ignored)
    """
    data = getlines(filename)
    contours = {}
    for line in data:
        if '>' in line[0]:
            # Is a header line, therefore split
            temp_vals = re.split(' ', line)
            contour_depth = int(abs(float(temp_vals[1])))
            #print contours, contour_depth, max_depth
            if (contour_depth > max_depth) or (contour_depth < min_depth):
                continue
             
            if not contour_depth in contours.keys():
             #   print contour_depth
                contours[contour_depth] = []
        else:
             temp_data = re.split('\t', (line).strip('\n'), 0)
             longitude = float(temp_data[0])
             if longitude > 180.:
                  longitude = -(360. - longitude)
             latitude = float(temp_data[1])
             depth = int(abs(float(temp_data[2])))
             #print depth
             if depth in contours.keys():
                 contours[depth].append(Point(longitude, 
                                              latitude, 
                                              float(depth)))
    edges = []
    contour_keys = contours.keys()
    contour_keys.sort()
    for key in contour_keys:
        edges.append(Line(contours[key]))
    return edges


def get_complex_interface(topfile, contour_file, max_depth=100, spacing=5.0):
    """
    Reads pair of Slab 1.0 top and contour files and returns the subduction
    interface in the form of an openquake.hazardlib.geo.ComplexFaultSurface
    class
    """
    # Load in top file
    edges = [read_upper_edge_file(topfile)]
    # Load in contours
    min_depth = np.min(np.array([point.depth for point in edges[0]]))
    int_edges = read_contour_file(contour_file, 1.,  max_depth)
    for conts in int_edges:
        min_edge_depth = np.min(np.array([point.depth for point in conts]))
        if min_edge_depth > min_depth:
            # Contour is deeper than previous edge - continue
            edges.append(conts)
            print min_edge_depth
        
    # Get geometry
    try:
        slab_fault = ComplexFaultSurface.from_fault_data(edges, spacing)
        return slab_fault, edges
    except:
        # Edges in wrong order - try flipping them
        #new_edges = []
        #for edge in edges:
        #    new_edges.append(Line(edge.points[::-1]))
        slab_fault = ComplexFaultSurface.from_fault_data(edges, spacing)
        return None, edges

def get_local_strikes_dips(interface):
    """
    From a subdction interface (defined as an instance of the class
    openquake.hazardlib.geo.ComplexFaultSurface) returns for each mesh point
    the local strike and dip.

    The complex interface defines the geomtry as three two-dimensional (Ns, Nd)
    arrays, each for longitude, latitude and depth. The function returns the
    local strike and dip in each cell and hence returns two (Ns - 1, Nd - 1)
    arrays.
    """
    plane_set = []
    ndd, nas = interface.mesh.shape
    azims = np.zeros_like(interface.mesh.lons)
    for iloc in xrange(nas):
        if iloc == 0:
            azims[:, iloc] = azimuth(interface.mesh.lons[:, iloc],
                                     interface.mesh.lats[:, iloc],
                                     interface.mesh.lons[:, iloc + 1],
                                     interface.mesh.lats[:, iloc + 1])
        elif iloc == (nas - 1):
            azims[:, iloc] = azimuth(interface.mesh.lons[:, iloc - 1],
                                     interface.mesh.lats[:, iloc - 1],
                                     interface.mesh.lons[:, iloc],
                                     interface.mesh.lats[:, iloc])
        else:
            azims[:, iloc] = azimuth(interface.mesh.lons[:, iloc - 1],
                                     interface.mesh.lats[:, iloc - 1],
                                     interface.mesh.lons[:, iloc + 1],
                                     interface.mesh.lats[:, iloc + 1])
    dips = np.zeros_like(interface.mesh.lons)
    for iloc in xrange(ndd):
        if iloc == 0:
            plane_dist = distance(interface.mesh.lons[iloc, :],
                                  interface.mesh.lats[iloc, :],
                                  interface.mesh.depths[iloc, :],
                                  interface.mesh.lons[iloc + 1, :],
                                  interface.mesh.lats[iloc + 1, :],
                                  interface.mesh.depths[iloc + 1, :])
            d_z = interface.mesh.depths[iloc + 1, :] -\
                interface.mesh.depths[iloc, :]

        elif iloc == (ndd - 1):
            plane_dist = distance(interface.mesh.lons[iloc - 1, :],
                                  interface.mesh.lats[iloc - 1, :],
                                  interface.mesh.depths[iloc - 1, :],
                                  interface.mesh.lons[iloc, :],
                                  interface.mesh.lats[iloc, :],
                                  interface.mesh.depths[iloc, :])
            d_z = interface.mesh.depths[iloc, :] -\
                interface.mesh.depths[iloc - 1, :]
        else:
            plane_dist = distance(interface.mesh.lons[iloc + 1, :],
                                  interface.mesh.lats[iloc + 1, :],
                                  interface.mesh.depths[iloc + 1, :],
                                  interface.mesh.lons[iloc - 1, :],
                                  interface.mesh.lats[iloc - 1, :],
                                  interface.mesh.depths[iloc - 1, :])
            d_z = interface.mesh.depths[iloc + 1, :] -\
                interface.mesh.depths[iloc - 1, :]
        dips[iloc, :] = np.degrees(np.arcsin(d_z / plane_dist))
    return azims, dips

            
def interface_framework(lons, lats, depths):
    """
    Define the lons and lats in terms of an along strike and down dip distance
    """
    ndd, nax = lons.shape
    xas = np.zeros_like(lons)
    ydd = np.zeros_like(lats)
    # Along strike
    for iloc in range(1, nax):
        xas[:, iloc] = xas[:, iloc - 1] + distance(
            lons[:, iloc - 1],
            lats[:, iloc - 1],
            depths[:, iloc - 1],
            lons[:, iloc],
            lats[:, iloc],
            depths[:, iloc])

    for iloc in range(1, ndd):
        ydd[iloc, :] = ydd[iloc - 1, :] + distance(
            lons[iloc - 1, :],
            lats[iloc - 1, :],
            depths[iloc - 1, :],
            lons[iloc, :],
            lats[iloc, :],
            depths[iloc, :])
    return xas, ydd

def interface_framework_2d(lons, lats, depths):
    """
    Define the lons and lats in terms of an along strike and across strike
    distance
    """
    ndd, nax = lons.shape
    xas = np.zeros_like(lons)
    ydd = np.zeros_like(lats)
    # Along strike
    for iloc in range(1, nax):
        xas[:, iloc] = xas[:, iloc - 1] + distance(
            lons[:, iloc - 1],
            lats[:, iloc - 1],
            np.zeros(len(depths[:, iloc - 1])),
            lons[:, iloc],
            lats[:, iloc],
            np.zeros(len(depths[:, iloc])))

    for iloc in range(1, ndd):
        ydd[iloc, :] = ydd[iloc - 1, :] + distance(
            lons[iloc - 1, :],
            lats[iloc - 1, :],
            np.zeros(len(depths[iloc - 1, :])),
            lons[iloc, :],
            lats[iloc, :],
            np.zeros(len(depths[iloc, :])))
    return xas, ydd


def get_strike_dxdy(dx, dy):
    """
    """
    if dx >= 0.:
        if dy >= 0.:
            # Upper Right quadrant
            return (pi / 2.) - atan(dy / dx)

        else:
            # Lower Right Quandrant
            return (pi / 2.) + atan(fabs(dy) / dx)
    else:
        if dy >= 0.:
            # Upper Left Quadrant
            return ((3. * pi) / 2.) + atan(dy / fabs(dx))

        else:
            # Lower Left Quadrant
            return pi + atan(fabs(dx) / fabs(dy))

def get_azimuth_dip(vect):
    """
    """
    cf = 180. / pi
    L = np.sqrt(np.sum(vect ** 2.))
    azimuth = atan(vect[0] / vect[1])
    dip = asin(-vect[2] / L)
    return azimuth * cf, fabs(dip * cf)


def get_azimuths_dips(fault):
    """
    """
    points, along_azimuth, updip, diag = fault.mesh.triangulate()


    # define planes that are perpendicular to each point's vector
    # as normals to those planes
    earth_surface_tangent_normal = geo_utils.normalized(points)

    # calculating triangles' area and normals for top-left triangles
    e1 = along_azimuth[:-1]
    e2 = updip[:, :-1]
    tl_area = geo_utils.triangle_area(e1, e2, diag)
    tl_normal = geo_utils.normalized(np.cross(e1, e2))
    # ... and bottom-right triangles
    e1 = along_azimuth[1:]
    e2 = updip[:, 1:]
    br_area = geo_utils.triangle_area(e1, e2, diag)
    br_normal = geo_utils.normalized(np.cross(e1, e2))
    # inclination calculation
    # top-left triangles
    en = earth_surface_tangent_normal[:-1, :-1]
    # cosine of inclination of the triangle is scalar product
    # of vector normal to triangle plane and (normalized) vector
    # pointing to top left corner of a triangle from earth center
    incl_cos = np.sum(en * tl_normal, axis=-1).clip(-1.0, 1.0)
    # we calculate average angle using mean of circular quantities
    # formula: define 2d vector for each triangle where length
    # of the vector corresponds to triangle's weight (we use triangle
    # area) and angle is equal to inclination angle. then we calculate
    # the angle of vector sum of all those vectors and that angle
    # is the weighted average.
    xx = np.sum(tl_area * incl_cos)
    # express sine via cosine using Pythagorean trigonometric identity,
    # this is a bit faster than sin(arccos(incl_cos))
    yy = np.sum(tl_area * np.sqrt(1 - incl_cos * incl_cos))

    # bottom-right triangles
    en = earth_surface_tangent_normal[1:, 1:]
    # we need to clip scalar product values because in some cases
    # they might exceed range where arccos is defined ([-1, 1])
    # because of floating point imprecision
    incl_cos = np.sum(en * br_normal, axis=-1).clip(-1.0, 1.0)
    # weighted angle vectors are calculated independently for top-left
    # and bottom-right triangles of each cell in a mesh. here we
    # combine both and finally get the weighted mean angle
    xx += br_area * incl_cos
    yy += br_area * np.sqrt(1 - incl_cos * incl_cos)
    inclination = np.degrees(np.arctan2(yy, xx))

    # azimuth calculation is done similar to one for inclination. we also
    # do separate calculations for top-left and bottom-right triangles
    # and also combine results using mean of circular quantities approach

    # unit vector along z axis
    z_unit = np.array([0.0, 0.0, 1.0])

    # unit vectors pointing west from each point of the mesh, they define
    # planes that contain meridian of respective point
    norms_west = geo_utils.normalized(np.cross(points + z_unit, points))
    # unit vectors parallel to planes defined by previous ones. they are
    # directed from each point to a point lying on z axis on the same
    # distance from earth center
    norms_north = geo_utils.normalized(np.cross(points, norms_west))
    # need to normalize triangles' azimuthal edges because we will project
    # them on other normals and thus calculate an angle in between
    along_azimuth = geo_utils.normalized(along_azimuth)

    # process top-left triangles
    # here we identify the sign of direction of the triangles' azimuthal
    # edges: is edge pointing west or east? for finding that we project
    # those edges to vectors directing to west by calculating scalar
    # product and get the sign of resulting value: if it is negative
    # than the resulting azimuth should be negative as top edge is pointing
    # west.
    sign = np.sign(np.sign(
        np.sum(along_azimuth[:-1] * norms_west[:-1, :-1], axis=-1))
        # we run numpy.sign(numpy.sign(...) + 0.1) to make resulting values
        # be only either -1 or 1 with zero values (when edge is pointing
        # strictly north or south) expressed as 1 (which means "don't
        # change the sign")
        + 0.1
    )
    # the length of projection of azimuthal edge on norms_north is cosine
    # of edge's azimuth
    az_cos = np.sum(along_azimuth[:-1] * norms_north[:-1, :-1], axis=-1)
    # use the same approach for finding the weighted mean
    # as for inclination (see above)
    xx = np.sum(tl_area * az_cos)
    # the only difference is that azimuth is defined in a range
    # [0, 360), so we need to have two reference planes and change
    # sign of projection on one normal to sign of projection to another one
    yy = np.sum(tl_area * np.sqrt(1 - az_cos * az_cos) * sign)

    # bottom-right triangles
    sign = np.sign(np.sign(
        np.sum(along_azimuth[1:] * norms_west[1:, 1:], axis=-1))
        + 0.1
    )
    az_cos = np.sum(along_azimuth[1:] * norms_north[1:, 1:], axis=-1)
    xx += br_area * az_cos
    yy += br_area * np.sqrt(1 - az_cos * az_cos) * sign
    azimuth = np.degrees(np.arctan2(yy, xx))
    idx = azimuth < 0
    if np.any(idx):
        azimuth[idx] += 360.


    idx = inclination > 90
    if np.any(idx):
        # average inclination is over 90 degree, that means that we need
        # to reverse azimuthal direction in order for inclination to be
        # in range [0, 90]
        inclination[idx] = 180. - inclination[idx]
        azimuth[idx] = (azimuth[idx] + 180) % 360
    return azimuth, inclination


def get_linestrings_from_mesh(lons, lats, depths):
    """
    From 2D arrays of longitudes, latitudes and depths returns a set of
    shapely.geo.LineStrings describing the along strike longitudes and
    latitudes, and the down-dip longitudes and latitudes
    """
    n_y, n_x = lons.shape
    down_transects = []
    along_transects = []
    for i in xrange(n_x):
        down_transects_lons.append(LineString([(lons[j, i], depths[j, i])
                                               for j in xrange(n_y)]))
        down_transects_lats.append(LineString([(lats[j, i], depths[j, i])
                                               for j in xrange(n_y)]))
    for i in xrange(n_y):
        along_transects_lons.append(LineString([(lons[i, j], depths[i, j])
                                           for j in xrange(n_x)]))
        along_transects_lats.append(LineString([(lats[i, j], depths[i, j])
                                           for j in xrange(n_x)]))
    return {"Down Longitude": down_transects_lons,
            "Down Latitude": down_transects_lats,
            "Along Longitude": along_transects_lons,
            "Along Latitude": along_transects_lats}

def get_linestrings_from_framework(asx, ydd, depths):
    """
    Builds a set of linestrings from along strike and down dip mesh
    """
    n_y, n_x = asx.shape
    down_transects = []
    along_transects = []
    for i in xrange(n_x):
        down_transects.append(LineString([(ydd[j, i], depths[j, i])
                                               for j in xrange(n_y)]))
    for i in xrange(n_y):
        along_transects.append(LineString([(asx[i, j], depths[i, j])
                                           for j in xrange(n_x)]))
    return along_transects, down_transects


def render_inslab_points(interface, thickness, depth_npd,
        layer_fractions=[0.5]):
    """
    Renders the inslab points
    """
    interface_mesh_points = []
    n_y, n_x = interface.mesh.shape
    transect_id = []
    lower_surface = np.zeros([n_y, n_x, 3])
    azimuth = (interface.get_strike() - 90.) % 360.
    dip_normal = 180. - (interface.get_dip() + 90.)
    inslab_points = []
    horiz_distance = thickness * np.cos(np.radians(dip_normal))
    vertical_distance = thickness * np.sin(np.radians(dip_normal))
    #print thickness, horiz_distance, vertical_distance
    for jloc in xrange(n_y):
        for iloc in xrange(n_x):
            mesh_point = Point(interface.mesh.lons[jloc, iloc],
                               interface.mesh.lats[jloc, iloc],
                               interface.mesh.depths[jloc, iloc])
            interface_mesh_points.append(mesh_point)
            transect_id.append((iloc, jloc))
            base_mesh_point = mesh_point.point_at(horiz_distance,
                                                  vertical_distance,
                                                  azimuth)
            lower_surface[jloc, iloc, 0] = base_mesh_point.longitude 
            lower_surface[jloc, iloc, 1] = base_mesh_point.latitude 
            lower_surface[jloc, iloc, 2] = base_mesh_point.depth 

    npds = []
    for fraction in layer_fractions:
        for pnt in interface_mesh_points:
            isp = pnt.point_at(horiz_distance, vertical_distance, azimuth)
            for (ud, ld), npd in depth_npd:
                if (isp.depth >= ud) and (isp.depth < ld):
                    npds.append(npd)
            inslab_points.append(pnt.point_at(horiz_distance,
                                              vertical_distance,
                                              azimuth))
    return inslab_points, npds, lower_surface, transect_id, azimuth


# By default the slab is considered totally impermeable
DEFAULT_POROSITY = [((0.0, 1000.), False)]

class InSlabSourceBuilder(object):
    """
    Tool to construct in-slab sources from an interface definition and
    return a set of OpenQuake point sources

    :param edges:
        List of edges (as openquake.hazardlib.geo.Line objects) defining the
        interface
    
    :param interface:
        Subduction interface geometry as instance of
        :class: openquake.hazardlib.geo.ComplexFaultSurface

    :param int ndd:
        Number of grid points down dip

    :param int nas:
        Number of grid points along strike

    :param npd_depths:
        ?

    :param numpy.ndarray asx:
        Along-strike distance (km) of the grid points

    :param numpy.ndarray nas:
        Down-dip distance (km) of the grid points

    :param list inslab_points:
        Resulting inslab point sources

    :param list npds:
        Nodal plane distribution as list of instances of :class:
        openquake.hazardlib.geo.NodalPlane

    :param list indices:
        ?

    :param numpy.ndarray stikes:
        Grid indicating the local strike of each grid cell

    :param numpy.ndarray dips:
        Grid indicating the local dip of each grid cell

    :param str npd_type:
        Indicates if the nodal planes are "absolute" (i.e. given in terms of
        aboslute strike and dip) or "relative" (i.e. given in terms of
        angle with respect to the strike and dip of the local interface cell)

    :param lower_surface:
        Surface indicating the lower side of the slab

    :param list porosity:
        List of values indicating whether a given depth range of the slab
        can be considered porous
    """
    def __init__(self, edges, spacing=10.):
        """
        Instantiate class with edges and spacing
        :param list edges:
            Subduction edges as list of instances of :class:
            openquake.hazardlib.geo.line.Line
        :param float spacing:
            Spacing (km)
        """
        # Builder
        self.edges = edges
        self.interface = ComplexFaultSurface.from_fault_data(edges, spacing)
        self.ndd, self.nas = self.interface.mesh.shape
        self.npd_depths = None
        self.asx, self.ddy = interface_framework_2d(self.interface.mesh.lons,
                                                    self.interface.mesh.lats,
                                                    self.interface.mesh.depths)
        self.inslab_points = []
        self.npds = []
        self.indices = []
        self.strikes, self.dips = get_local_strikes_dips(self.interface)
        self.npd_type = None
        self.lower_surface = None
        self.porosity = []
        self.sources = []

    @classmethod
    def from_contour_files(cls, top_filename, contour_filename, max_depth,
            spacing=10.):
        """
        Builds the interface from Slab 1.0contour files
        :param str top_filename:
            File of top edge contour
        :param str contour_filename:
            File containing remaining contours
        :param float max_depth:
            Maximum depth to consider
        :param float spacing:
            Mesh spacing
        """
        # Load in top file
        edges = [read_upper_edge_file(top_filename)]
        # Load in contours
        min_depth = np.min(np.array([point.depth for point in edges[0]]))
        print "Minimum Depth (km) = %.4f" % min_depth
        int_edges = read_contour_file(contour_filename, 1.,  max_depth)
        for conts in int_edges:
            min_edge_depth = np.min(np.array([point.depth for point in conts]))
            if min_edge_depth > min_depth:
                # Contour is deeper than previous edge - continue
                edges.append(conts)
                print "---- Contour Depth (km) %.4f" % min_edge_depth
        
        return cls(edges, spacing)

    def __len__(self):
        return len(self.inslab_points)
    
    def get_number_sources(self):
        """
        Returns the number of sources
        """
        return len(self.inslab_points)

    def render_inslab_points(self, thickness, depth_npd, layer_fractions=[0.5],
            porosity_distribution=DEFAULT_POROSITY, npd_type="Absolute"):
        """
        Renders the inslab points projecting the interface vertically downward
        :param float thickness:
            Thickness of the slab (km)

        :param list depth_npd:
            Nodal plane distribution with depth as list of tuples where the
            tuple is defined as
            [((upper_depth_limit_1, lower_depth_limit_1), PMF_1),
             ((upper_depth_limit_2, lower_depth_limit_2), PMF_2),
             ...
             ((upper_depth_limit_n, lower_depth_limit_n), PMF_3)]

        :param list layer_fractions:
             List of locations as fractions of the slab thickness to place the
             layers of points

        :param list porosity_distribution:
             List of
            [((upper_depth_limit_1, lower_depth_limit_1), False),
             ((upper_depth_limit_2, lower_depth_limit_2), False),
             ...
             ((upper_depth_limit_n, lower_depth_limit_n), True)]
             
        :param str npd_type:
            Indicates whether the nodal plane distribution is "Relative"
            (strikes and dips as angles with respect to the interface) or
            absolute (strikes and dips as conventional angles)
        """
        self.npd_type = npd_type
        interface_mesh_points = []
        n_y, n_x = self.interface.mesh.shape
        depths = np.zeros_like(self.interface.mesh.lons)
        self.lower_surface = RectangularMesh(
            self.interface.mesh.lons,
            self.interface.mesh.lats,
            self.interface.mesh.depths + thickness)

        for fraction in layer_fractions:
            for iloc in xrange(self.ndd):
                for jloc in xrange(self.nas):
                    isp = Point(self.interface.mesh.lons[iloc, jloc],
                                self.interface.mesh.lats[iloc, jloc],
                                self.interface.mesh.depths[iloc, jloc] +
                                fraction * thickness)
                    for (ud, ld), npd in depth_npd:
                        if (isp.depth >= ud) and (isp.depth < ld):
                            self.npds.append(npd)
                    for (ud, ld), porosity in porosity_distribution:
                        if (isp.depth >= ud) and (isp.depth < ld):
                            self.porosity.append(porosity)
                    self.inslab_points.append(isp)
                    self.indices.append((iloc, jloc))

    def render_inslab_points_normal(self, thickness, depth_npd,
            layer_fractions=[0.5], porosity_distribution=DEFAULT_POROSITY,
            npd_type="Absolute"):
        """
        Renders the inslab points projecting the slab perpendicular to the
        interface
        """
        self.npd_type = npd_type
        n_y, n_x = self.interface.mesh.shape
        depths = np.zeros_like(self.interface.mesh.lons)
        self.lower_surface = np.zeros([n_y, n_x, 3])
        azimuth = (self.interface.get_strike() - 90.) % 360.
        dip_normal = 180. - (self.interface.get_dip() + 90.)
        horiz_distance = thickness * np.cos(np.radians(dip_normal))
        vertical_distance = thickness * np.sin(np.radians(dip_normal))
        #print thickness, horiz_distance, vertical_distance

        for iloc in xrange(self.ndd):
            for jloc in xrange(self.nas):
                
                isp = Point(self.interface.mesh.lons[iloc, jloc],
                            self.interface.mesh.lats[iloc, jloc],
                            self.interface.mesh.depths[iloc, jloc])
                base_mesh_point = isp.point_at(horiz_distance,
                                               vertical_distance,
                                               azimuth)
                self.lower_surface[iloc, jloc, 0] = base_mesh_point.longitude
                self.lower_surface[iloc, jloc, 1] = base_mesh_point.latitude
                self.lower_surface[iloc, jloc, 2] = base_mesh_point.depth
        self.lower_surface = RectangularMesh(self.lower_surface[:, :, 0],
                                             self.lower_surface[:, :, 1],
                                             self.lower_surface[:, :, 2])
        
        for fraction in layer_fractions:
            for iloc in xrange(self.ndd):
                for jloc in xrange(self.nas):
                    isp = Point(self.interface.mesh.lons[iloc, jloc],
                                self.interface.mesh.lats[iloc, jloc],
                                self.interface.mesh.depths[iloc, jloc])
                    for (ud, ld), npd in depth_npd:
                        if (isp.depth >= ud) and (isp.depth < ld):
                            self.npds.append(npd)
                    for (ud, ld), porosity in porosity_distribution:
                        if (isp.depth >= ud) and (isp.depth < ld):
                            self.porosity.append(porosity)
                    self.inslab_points.append(
                        isp.point_at(fraction * horiz_distance,
                                     fraction * vertical_distance,
                                     azimuth))
                    self.indices.append((iloc, jloc))

    def build_source_model(self, mfd, id_stem, msr=StrasserIntraslab(),
            usd=0.0, lsd=1000.0, aspect=1.5, rupture_mesh_spacing=1.0):
        """
        Constructs the source model subject to the constraints previously
        defined
        :param mfd:
            Magnitude frequency distribution as instance of the :class:
            openquake.hazardlib.mfd.base.BaseMFD
        :param str id_stem:
            Stem to attach to ID
        :param msr:
            Magnitude scaling relation as instance of :class:
            openquake.hazardlib.scalerel.base.BaseMSR
        :param float usd:
            Upper seismogenic depth (global value)
        :param float lsd:
            Lower seismogenic depth (global value)
        :param float aspect:
            Aspect ratio
        :param float rupture_mesh_spacing:
            Rupture mesh specing (km) - Not used in export

        """
        _, ddip_lines_surface = get_linestrings_from_framework(
            self.asx,
            self.ddy,
            self.interface.mesh.depths)
        _, ddip_lines_base = get_linestrings_from_framework(
            self.asx,
            self.ddy,
            self.lower_surface.depths)
        tom = PoissonTOM(1.0)
        mags, annual_rates = zip(*mfd.get_annual_occurrence_rates())
        annual_rates = np.array(annual_rates)
        point_rate = annual_rates / float(self.get_number_sources())
        self.sources = []
        for iloc in xrange(self.get_number_sources()):
            point_location = self.inslab_points[iloc]
            idxi, idxj = self.indices[iloc]

            if self.npd_type == "Relative":
                # Strike and dip are given with respect to the interface
                source_npd = []
                #source_rates = []
                for prob, npd in self.npds[iloc].data:
                    if_strike = self.strikes[idxi, idxj]
                    if_dip = self.dips[idxi, idxj]
                    is_strike = (if_strike + npd.strike) % 359.9
                    is_dip = if_dip + npd.dip
                    if is_dip > 90.:
                        is_dip = 180.0 - is_dip
                    new_npd = NodalPlane(round(is_strike, 1),
                                         round(is_dip, 1), 
                                         round(npd.rake, 1))
                    source_npd.append([prob, new_npd])
                #    source_rates.append(annual_rates * prob)
                source_npd = PMF(source_npd)
            elif self.npd_type == "Absolute":
                source_npd = self.npds[iloc]

            else:
                raise ValueError("Nodal plane type not supported!")
            hdd = PMF([(1.0, point_location.depth)])
            if self.porosity[iloc]:
                # Both interfaces are porous
                point_mfd = EvenlyDiscretizedMFD(mfd.min_mag,
                                                 mfd.bin_width,
                                                 point_rate.tolist())
                point_source = PointSource(
                    "{:s}_{:s}".format(id_stem, str(iloc)),
                    "PNT_{:s}".format(str(iloc)),
                    "Subduction IntraSlab",
                    point_mfd,
                    rupture_mesh_spacing,
                    msr,
                    aspect,
                    tom,
                    usd,
                    lsd,
                    point_location,
                    source_npd,
                    hdd)
                self.sources.append(point_source)
            else:
                point_set, point_mfd_set, upper_depths, lower_depths =\
                    self._get_upper_lower_depths(point_location, 
                                                 ddip_lines_surface,
                                                 ddip_lines_base,
                                                 point_rate,
                                                 idxi, idxj, source_npd)
                point_sources = []
                id_counter = 0
                for jloc, new_point in enumerate(point_set):
                    point_mfd = EvenlyDiscretizedMFD(mfd.min_mag,
                                                     mfd.bin_width,
                                                     point_mfd_set[jloc].tolist())
                    pnt_id = "{:s}_{:s}_{:s}".format(id_stem,
                                                     str(iloc),
                                                     str(id_counter))
                    pnt_name = "PNT_{:s}-{:s}".format(str(iloc),
                                                      str(id_counter))
                    #print pnt_id, pnt_name
                    source = PointSource(pnt_id,
                                         pnt_name,
                                         "Subduction IntraSlab",
                                         point_mfd,
                                         rupture_mesh_spacing,
                                         msr,
                                         aspect,
                                         tom,
                                         upper_depths[jloc],
                                         lower_depths[jloc],
                                         new_point,
                                         source_npd,
                                         hdd)
                    self.sources.append(source)
                    id_counter += 1

    def get_sources_leaky(self, mfd, id_stem, msr=StrasserIntraslab(),
            usd=0.0, lsd=1000.0, aspect=1.5, rupture_mesh_spacing=1.0):
        """
        Builds the source models whose depths would not be constrained by
        the limits of the subduction zones
        """

        tom = PoissonTOM(1.0)
        mags, annual_rates = zip(*mfd.get_annual_occurrence_rates())
        annual_rates = np.array(annual_rates)
        point_rate = annual_rates / float(self.get_number_sources())
        self.sources = []
        for iloc in xrange(self.get_number_sources()):
            point_location = self.inslab_points[iloc]
            idxi, idxj = self.indices[iloc]

            if self.npd_type == "Relative":
                # Strike and dip are given with respect to the interface
                source_npd = []
                #source_rates = []
                for prob, npd in self.npds[iloc].data:
                    if_strike = self.strikes[idxi, idxj]
                    if_dip = self.dips[idxi, idxj]
                    is_strike = (if_strike + npd.strike) % 359.9
                    is_dip = if_dip + npd.dip
                    if is_dip > 90.:
                        is_dip = 180.0 - is_dip
                    new_npd = NodalPlane(round(is_strike, 1),
                                         round(is_dip, 1),
                                         round(npd.rake))
                    source_npd.append([prob, new_npd])
                source_npd = PMF(source_npd)
            elif self.npd_type == "Absolute":
                source_npd = self.npds[iloc]

            else:
                raise ValueError("Nodal plane type not supported!")
            hdd = PMF([(1.0, point_location.depth)])
            point_mfd = EvenlyDiscretizedMFD(mfd.min_mag,
                                             mfd.bin_width,
                                             point_rate.tolist())
            point_source = PointSource("{:s}_{:s}".format(id_stem, str(iloc)),
                                       "PNT_{:s}".format(str(iloc)),
                                       "Subduction IntraSlab",
                                       point_mfd,
                                       1.0,
                                       msr,
                                       aspect,
                                       tom,
                                       usd,
                                       lsd,
                                       point_location,
                                       source_npd,
                                       hdd)
            self.sources.append(point_source)

    def get_sources_constrained(self, mfd, id_stem, msr=StrasserIntraslab(),
            aspect=1.5, rupture_mesh_spacing=1.0):
        """
        Builds the source model with the upper and lower depths constrained
        by the interface and lower surface
        """
        _, ddip_lines_surface = get_linestrings_from_framework(
            self.asx,
            self.ddy,
            self.interface.mesh.depths)
        _, ddip_lines_base = get_linestrings_from_framework(
            self.asx,
            self.ddy,
            self.lower_surface.depths)

        
        tom = PoissonTOM(1.0)
        mags, annual_rates = zip(*mfd.get_annual_occurrence_rates())
        annual_rates = np.array(annual_rates)
        point_rate = annual_rates / float(self.get_number_sources())
        self.sources = []
        id_counter = 0
        for iloc in xrange(self.get_number_sources()):
            point_location = self.inslab_points[iloc]
            idxi, idxj = self.indices[iloc]

            if self.npd_type == "Relative":
                # Strike and dip are given with respect to the interface
                source_npd = []
                for prob, npd in self.npds[iloc].data:
                    if_strike = self.strikes[idxi, idxj]
                    if_dip = self.dips[idxi, idxj]
                    is_strike = (if_strike + npd.strike) % 359.9
                    is_dip = if_dip + npd.dip
                    if is_dip > 90.:
                        is_dip = 180.0 - is_dip
                        # Dipping in the other direction - flip strike!
                        is_strike = (is_strike + 180.) % 359.9
                    new_npd = NodalPlane(is_strike, is_dip, npd.rake)
                    source_npd.append([prob, new_npd])
                #    source_rates.append(annual_rates * prob)
                source_npd = PMF(source_npd)
            elif self.npd_type == "Absolute":
                source_npd = self.npds[iloc]

            else:
                raise ValueError("Nodal plane type not supported!")
            hdd = PMF([(1.0, point_location.depth)])
            point_set, point_mfd_set, upper_depths, lower_depths =\
                self._get_upper_lower_depths(point_location, 
                                             ddip_lines_surface,
                                             ddip_lines_base,
                                             point_rate,
                                             idxi, idxj, source_npd)
            point_sources = []
            for jloc, new_point in enumerate(point_set):
                point_mfd = EvenlyDiscretizedMFD(mfd.min_mag,
                                                 mfd.bin_width,
                                                 point_mfd_set[jloc].tolist())
                pnt_id = "{:s}_{:s}_{:s}".format(id_stem,
                                                 str(iloc),
                                                 str(jloc))
                pnt_name = "PNT_{:s}_{:s}".format(str(iloc), str(jloc))
                #print pnt_id, pnt_name
                source = PointSource(pnt_id,
                                     pnt_name,
                                     "Subduction IntraSlab",
                                     point_mfd,
                                     1.0,
                                     msr,
                                     aspect,
                                     tom,
                                     upper_depths[jloc],
                                     lower_depths[jloc],
                                     new_point,
                                     source_npd,
                                     hdd)
                self.sources.append(source)
                id_counter += 1

    def _get_upper_lower_depths(self, point, ddip_lines_surface,
            ddip_lines_base, point_rates, idxi, idxj, npds):
        """
        Returns the upper and lower depths associated with each point
        """
        point_set = []
        point_mfd_set = []
        point_upper_depths = []
        point_lower_depths = []
        centre_point_as = self.asx[idxi, idxj]
        centre_point_dd = self.ddy[idxi, idxj]
        #print point
        for prob, npd in npds.data:
            # Get up-dip lengthmodel
            up_dip_length = point.depth * np.tan(np.radians(90.0 - npd.dip))
            down_dip_length = (1000.0 - point.depth) /\
                np.tan(np.radians(npd.dip))
            # Get horizontal length
            rad_strike = np.radians(npd.strike - self.strikes[idxi, idxj])
            along_length_up = up_dip_length / np.cos(rad_strike)
            along_length_down = down_dip_length / np.cos(rad_strike)
            # Build linestring
            top_point = (centre_point_dd - along_length_up,
                         0.0)

            bottom_point = (centre_point_dd + along_length_down, 1000.0)
            fault_plane = LineString([top_point, bottom_point])
            
            upper_depth, lower_depth = self._check_point_upper(
                top_point,
                fault_plane,
                idxi,
                ddip_lines_surface[idxj],
                ddip_lines_base[idxj])
            if upper_depth > point.depth:
                upper_depth = self.interface.mesh.depths[idxi, idxj]
            if lower_depth < point.depth:
                lower_depth = self.lower_surface.depths[idxi, idxj]

            # Surface intersection
            point_upper_depths.append(upper_depth)
            point_lower_depths.append(lower_depth)
            point_set.append(point)
            point_mfd_set.append(point_rates * prob)
        return point_set, point_mfd_set, point_upper_depths, point_lower_depths

    def _check_point_upper(self, top_point, fault_plane, idxi, surface_line,
            base_line):
        """
        Checks to see if plane would intersect with the upper and lower
        surfaces
        """
        isctn_u = surface_line.intersection(fault_plane)
        isctn_l = base_line.intersection(fault_plane)
        # Upper depth
        if top_point[0] < 0.0 or isctn_u.is_empty:
            # Does not intersect upper plane
            upper_depth = self.interface.mesh.depths[idxi, 0]
        elif isinstance(isctn_u, geometry.MultiPoint):
            # Crosses in multiple places
            upper_depth = isctn_u[0].y
        else:
            # Is a single point
            upper_depth = isctn_u.y

        # Lower Depth
        if isinstance(isctn_u, geometry.MultiPoint):
            # Cuts the upper plane in multiple places 
            lower_depth = isctn_u[1].y
        elif isctn_l.is_empty:
            # No intersection (theoretically no lower limit)
            lower_depth = 300.0
        elif isinstance(isctn_l, geometry.MultiPoint):
            lower_depth = isctn_l[0].y
        else:
            # Single intersection
            lower_depth = isctn_l.y
        return upper_depth, lower_depth

    def write_source_model(self, output_filename, source_model_name=None):
        """
        Exports the source model to xml
        :param str output_filename:
            Name of file for writing
        :param str source_model_name:
            Name of output source model
        """
        if not len(self.sources):
            raise ValueError("Cannot write empty source model")
        write_source_model(output_filename, self.sources,
                           source_model_name)
