import numpy as np

try:
    import controllers.perception_controller.CONSTANTS as CONSTANTS
except ImportError:
    import CONSTANTS as CONSTANTS


def find_viewing_distance(
    card_size: float = 0.06, focal_length: int = 80, camera_pixels: int = 32
) -> float:
    """
    Given the focal length of a camera, and the size of a seat-number card in the world,
    this function finds the viewing distance required for the card to exactly fill the image,
    assuming that the camera's optical axis is orthogonal to the card
    (the image plane is parallel to the card).

    Parameters
    ---------
    card_size : float, default 0.06 m
        Side length of the (square) seat card in the world in meters.
        Card is a square card_size x card_size. Default is 6cmx6cm.
    focal_length : int, default 80 pixels
        Focal length of the pinhole camera in pixels. Default is 80 pixels.
    camera_pixels : int, default 32 pixels
        Number of pixels in the camera (square) width and height. Default is 32 x 32 pixels.

    Returns
    -------
    float
        The distance (in meters) from the camera to the card face at
        which the card exactly fills the image.
    """
    # for card to fill image, image size should equal camera_pixels
    viewing_distance = focal_length * card_size / camera_pixels  # pixels * m / pixels = m
    return viewing_distance


def find_viewing_location_world(
    card_world_translation: np.ndarray,
    normal_vector: np.ndarray,
    viewing_distance: float,
    card_thickness: float,
) -> tuple[np.ndarray, float]:
    """
    Given the translation of a seat card in the world (the center of the card), and the
    normal vector out of the card face, finds the location in the world the robot would have
    to drive to get a clean picture of the card.

    The card has a thickness is not accounted for in its translation
    (the center of the card), so function adds half of the thickness of the card along
    the normal vector. Some guests left their stools pushed out at an angle, so the normal
    vector is not always aligned with the x or y axis.

    Parameters
    ----------
    card_world_translation : np.ndarray
        Translation of the card center in the world.
    normal_vector : np.ndarray
        Normal vector out of the card face, normalized to length 1, in the world frame.
    viewing_distance : float
        Distance from the camera to the card face.
    card_thickness : float
        Thickness of the card in the world, in meters.

    Returns
    -------
    tuple[np.ndarray, float]
        translation and yaw in radians the robot would have to drive to
        between -180 and 180 degrees.
        The yaw here refers to the yaw of the robot's heading in the world frame.
    """
    card_x = card_world_translation[0] + normal_vector[0] * card_thickness / 2
    card_y = card_world_translation[1] + normal_vector[1] * card_thickness / 2
    viewing_dist_x = normal_vector[0] * viewing_distance
    viewing_dist_y = normal_vector[1] * viewing_distance

    x = card_x + viewing_dist_x
    y = card_y + viewing_dist_y

    # robot's yaw should be exact opposite of card's, so it faces it
    yaw = np.arctan2(-normal_vector[1], -normal_vector[0])

    return (np.array([x, y]), yaw)


def get_all_viewing_locations_world():
    """
    Uses the two functions above to find the viewing location (and yaw) of every seat card.
    """
    card_thickness = CONSTANTS.CARD_THICKNESS_METERS
    card_size = CONSTANTS.CARD_SIZE_METERS
    focal_length = CONSTANTS.FOCAL_LENGTH_PIXELS
    camera_pixels = CONSTANTS.CAMERA_PIXELS
    seat_info_dict = CONSTANTS.SEAT_INFO
    viewing_locations_world = {}
    for seat_index in seat_info_dict:
        card_translation, card_normal_vector = seat_info_dict[seat_index]
        viewing_distance = find_viewing_distance(card_size, focal_length, camera_pixels)
        viewing_location_world, yaw = find_viewing_location_world(
            card_translation, card_normal_vector, viewing_distance, card_thickness
        )
        viewing_locations_world[seat_index] = (viewing_location_world, yaw)
    return viewing_locations_world
