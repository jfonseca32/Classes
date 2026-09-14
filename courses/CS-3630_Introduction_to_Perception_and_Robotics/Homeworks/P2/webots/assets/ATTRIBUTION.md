# CAD mesh attribution

The sourced meshes in this folder were centered, uniformly rescaled,
reoriented where needed, and re-exported as binary STL files for reliable
Webots rendering. The wing mesh is original course-project geometry as noted
below.

## Cyberbotics Webots assets

The world uses Cyberbotics' official ABB IRB 4600/40, factory floor and
ceiling, walls, conveyor belt, cabinet, pallet stack, cardboard box, and oil
barrel PROTOs from the Webots R2025a repository. The ABB PROTO and its seven
OBJ meshes are included locally under `../protos/` so the robot does not depend
on a GitHub download at runtime. The decorative factory PROTOs continue to use
their official Cyberbotics URLs. These assets remain subject to the
[Webots assets license](https://cyberbotics.com/webots_assets_license) and are
licensed for use with Webots.

## `printed_product.stl` and `structural_brace.stl`

- Source: [CadQuarry demo dataset](https://github.com/jacobjennings/CadQuarry)
- Original files: `gear_498d670f_0025.stl` and
  `bracket_498d6701_0011.stl`
- Data license: [CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/)

## `pcb_board.stl`

- Source: [SparkFun 3D Models](https://github.com/sparkfun/3D_Models)
- Original file: `SPARK_PHOTON_TEMPLATE_WIDE.stl`
- Copyright: SparkFun Electronics
- Hardware-model license:
  [Creative Commons Attribution-ShareAlike 4.0](https://creativecommons.org/licenses/by-sa/4.0/)
- Changes: centered, uniformly rescaled, and re-exported as binary STL. This
  adapted mesh remains available under CC BY-SA 4.0.

## `wing.stl`

- Source: generated specifically for this CS 3630 Webots visualization
- Geometry: tapered, mildly swept wing using a cambered NACA 2412 airfoil
  section
- License: original course-project asset; no third-party CAD source

## `motor.stl`

- Source: [SoftRafts](https://github.com/SMILE-Robotics-Lab/SoftRafts)
- Original file: `CAD/Motor.stl`
- License: MIT
- Copyright (c) 2025 Luyang Zhao

## MIT license text

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
