# Asset attribution

The Project 3 world (`webots/worlds/project3_vacuum.wbt`) is built from
official Cyberbotics Webots R2025a assets plus nine generated label textures.
Nothing else is sourced from third parties.

## Cyberbotics Webots R2025a PROTOs (fetched at load time)

The world pins every PROTO to the R2025a tag on GitHub, so the scene is
identical on every machine and is cached by Webots after the first load:

| PROTO | URL |
|---|---|
| `Floor` | https://raw.githubusercontent.com/cyberbotics/webots/R2025a/projects/objects/floors/protos/Floor.proto |
| `Parquetry` (appearance) | https://raw.githubusercontent.com/cyberbotics/webots/R2025a/projects/appearances/protos/Parquetry.proto |
| `Wall` | https://raw.githubusercontent.com/cyberbotics/webots/R2025a/projects/objects/apartment_structure/protos/Wall.proto |
| `CeilingLight` | https://raw.githubusercontent.com/cyberbotics/webots/R2025a/projects/objects/lights/protos/CeilingLight.proto |
| `Desk` | https://raw.githubusercontent.com/cyberbotics/webots/R2025a/projects/objects/tables/protos/Desk.proto |
| `OfficeChair` | https://raw.githubusercontent.com/cyberbotics/webots/R2025a/projects/objects/chairs/protos/OfficeChair.proto |
| `Monitor` | https://raw.githubusercontent.com/cyberbotics/webots/R2025a/projects/objects/computers/protos/Monitor.proto |
| `Fridge` | https://raw.githubusercontent.com/cyberbotics/webots/R2025a/projects/objects/kitchen/fridge/protos/Fridge.proto |
| `Oven` | https://raw.githubusercontent.com/cyberbotics/webots/R2025a/projects/objects/kitchen/oven/protos/Oven.proto |
| `Table` | https://raw.githubusercontent.com/cyberbotics/webots/R2025a/projects/objects/tables/protos/Table.proto |
| `Toilet` | https://raw.githubusercontent.com/cyberbotics/webots/R2025a/projects/objects/bathroom/protos/Toilet.proto |
| `BathroomSink` | https://raw.githubusercontent.com/cyberbotics/webots/R2025a/projects/objects/bathroom/protos/BathroomSink.proto |
| `StraightStairs` (+ its `StraightStairsRail` dependency) | https://raw.githubusercontent.com/cyberbotics/webots/R2025a/projects/objects/stairs/protos/StraightStairs.proto |

`Parquetry` is released under the Apache License 2.0
(https://www.apache.org/licenses/LICENSE-2.0). All the other PROTOs above
carry the header "Copyright Cyberbotics Ltd. Licensed for use only with
Webots" and remain subject to the
[Webots assets license](https://cyberbotics.com/webots_assets_license); they
are used here only inside Webots. The textures they reference are downloaded
by Webots from the same R2025a tree.

## Vendored iRobot Create (`webots/protos/Create.proto` + `webots/protos/textures/`)

- Source: https://raw.githubusercontent.com/cyberbotics/webots/R2025a/projects/robots/irobot/create/protos/Create.proto
  and its four textures `create_base_color.jpg`, `create_normal.jpg`,
  `create_occlusion.jpg`, `create_roughness.jpg` from
  https://github.com/cyberbotics/webots/tree/R2025a/projects/robots/irobot/create/protos/textures
- License: Copyright Cyberbotics Ltd., licensed for use only with Webots
  (https://cyberbotics.com/webots_assets_license). The original license header
  is kept at the top of the file.
- Why vendored: the model must be kinematic (no ODE bodies) so the Supervisor
  can move it by writing its `translation`/`rotation` fields along authored
  waypoints without contact jitter.
- Modifications (each listed with the original line numbers in a comment block
  directly under the license header of the vendored file): removed the
  robot-level `physics Physics {...}` and `boundingObject Cylinder {...}`;
  removed `boundingObject`/`physics` from the two driven wheel `Solid`s, the
  two passive wheel `Solid`s, and the two bumper `TouchSensor`s. Every shape,
  texture reference, `HingeJoint`, `RotationalMotor`, `PositionSensor`, `LED`,
  `Receiver`, `DistanceSensor` and the `bodySlot` field are unchanged, so the
  wheels can still be spun by the controller for a visual effect. The
  `LightSensor` and `Pen` used by the assignment are added from the world file
  through `bodySlot`, not inside the PROTO.

## Generated room labels (`webots/assets/room_labels/room_1.png` ... `room_9.png`)

- Source: generated for this course project by
  `instructor/tools/make_room_labels.py` (Pillow; room number and short room
  name on a white tile, 256 x 256 pixels).
- License: original course-project asset; no third-party source. The font is
  whichever bold system font Pillow finds first (Arial Bold / Helvetica /
  DejaVu Sans Bold) or Pillow's built-in fallback; no font file is shipped.
