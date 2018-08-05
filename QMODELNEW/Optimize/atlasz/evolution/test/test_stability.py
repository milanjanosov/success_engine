"""This is a quick test to test the stability algo."""

# external imports
import os
import sys
import numpy

try:
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
except ImportError:
    # No matplotlib, meh.
    matplotlib = None


#internal imports
print(sys.modules[__name__].__file__)
sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(sys.modules[__name__].__file__), "../..")))
sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(sys.modules[__name__].__file__), "../../..")))
import evolution.stability


def main(argv=[]):
    """Main entry point."""
    dim = 13
    n = 10000
    center = [0.5]*dim
    numpy.random.seed(12345)
    points = [[numpy.random.random() for i in range(dim)] for j in range(n)]
    r = evolution.stability.get_largest_empty_cube_around_center(center, points)
    print "largest_empty_cube_around_center:", r
    minmax = evolution.stability.get_largest_empty_volume_around_center(center, points, r)
    print "largest_empty_volume_around_center:", minmax
    if matplotlib is None:
        return

    fig, ax = plt.subplots()
    plt.plot([center[0]], [center[1]], 'ro')
    plt.plot([points[i][0] for i in range(n)], [points[i][1] for i in range(n)], 'bo')
    patches = []
    ax.add_patch(Rectangle(xy=[center[0]-r, center[1]-r], width=2*r, height=2*r,
            facecolor='none', edgecolor='k'))
    ax.add_patch(Rectangle(
            xy=[minmax[0][0], minmax[1][0]],
            width=minmax[0][1] - minmax[0][0],
            height=minmax[1][1] - minmax[1][0],
            facecolor='none', edgecolor='k'))
    plt.axis('equal')
    plt.show()


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:])) # pass only real params to main
    except Exception as ex:
        print >>sys.stderr, ex
        import traceback
        traceback.print_exc(ex)
        sys.exit(1)
