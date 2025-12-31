# Model Generation

## Prerequisites

### Install Docker

Before you begin, you need Docker installed on your computer. Docker allows you to run the terrain generator without installing complex dependencies.

**Check if Docker is already installed:**

Open your terminal (on Mac: search for "Terminal" in Spotlight; on Linux: search for "Terminal" in your applications) and type:

```bash
docker --version
```

If you see a version number (like `Docker version 24.0.0`), Docker is installed and you can skip to the next section.

**If Docker is not installed:**

- **Mac**: Download and install [Docker Desktop for Mac](https://docs.docker.com/desktop/setup/install/mac-install/)
- **Linux**: Run this command in your terminal:
  ```bash
  curl -fsSL https://get.docker.com | sudo sh
  ```
  See the [official Docker installation guide](https://docs.docker.com/engine/install/debian/#install-using-the-convenience-script) for more details.

After installing, restart your terminal and verify with `docker --version`.

---

## Step 1: Create a Working Directory

First, create a folder on your computer where all the files will be stored. Open your terminal and run these commands one at a time:

```bash
# Create a new folder called "terrain-project" in your home directory
mkdir -p ~/terrain-project

# Navigate into that folder
cd ~/terrain-project

# Create subfolders for configs, elevation data, and outputs
mkdir -p configs topo outputs
```

**What this does:**
- `mkdir -p` creates new folders
- `cd` changes your current directory
- `~/` refers to your home folder (e.g., `/Users/yourname/` on Mac)

---

## Step 2: Create a Configuration File

The configuration file tells the generator what area of the world to create a model of. We'll use an interactive wizard to create it.

### Find Your Coordinates

Before running the wizard, you need to know the geographic coordinates (latitude and longitude) of the area you want to model.

1. Go to [bboxfinder.com](http://bboxfinder.com)
2. Navigate to the area you want to model
3. Draw a rectangle around your area by clicking and dragging
4. Look at the **Box** coordinates at the bottom of the screen
5. Note down the four numbers - they represent:
   - **Min Longitude** (west edge) - the first number, negative for locations in the Americas
   - **Min Latitude** (south edge) - the second number
   - **Max Longitude** (east edge) - the third number
   - **Max Latitude** (north edge) - the fourth number

**Example:** For downtown San Francisco, the coordinates might be:
- Min Longitude: `-122.42`
- Min Latitude: `37.77`
- Max Longitude: `-122.38`
- Max Latitude: `37.80`

### Run the Configuration Wizard

Now run this command in your terminal (make sure you're still in the `~/terrain-project` folder):

```bash
bash <(curl -fsSL https://raw.githubusercontent.com/rdudhagra/topo-tessellate/main/create-config.sh) configs/my_terrain.yaml
```

**What this does:**
- Downloads and runs the configuration wizard
- Creates a file called `my_terrain.yaml` in your `configs` folder

The wizard will ask you several questions. Here's what each one means:

| Question | What to Enter |
|----------|---------------|
| **Project name** | A name for your project (e.g., "San Francisco Downtown") |
| **Output prefix** | A short name for output files (e.g., "sf_downtown") |
| **Min/Max Longitude/Latitude** | The coordinates you found on bboxfinder.com |
| **Maximum model length** | Size of the longest side in millimeters (225 is good for 3D printing) |
| **Base height** | Thickness of the base in mm (20 is recommended minimum) |
| **Elevation multiplier** | Vertical exaggeration (1 = real scale, 2 = double height) |
| **Downsample factor** | Resolution reduction (1 = highest detail, 10 = faster/smaller files) |
| **Enable buildings** | Whether to include 3D buildings from OpenStreetMap (y/n) |
| **Tile rows** | Number of rows to split model into (1 = no splitting) |
| **Tile columns** | Number of columns to split model into (1 = no splitting) |
| **Output directory** | Where to save the generated files |

Press **Enter** to accept the default value shown in brackets.

---

## Step 3: Download Elevation Data

Now you need to download the elevation data for your area. This data comes from the USGS (United States Geological Survey) and is free to use.

Run this command in your terminal:

```bash
bash <(curl -fsSL https://raw.githubusercontent.com/rdudhagra/topo-tessellate/main/download-dem.sh) \
    --config configs/my_terrain.yaml \
    --topo-dir ./topo
```

**What this does:**
- Connects to the USGS National Map servers
- Downloads high-resolution elevation data for your area
- Saves the files to your `topo` folder

**Note:** This may take several minutes depending on the size of your area and your internet connection. The files can be quite large (hundreds of megabytes for detailed areas).

You'll see progress information showing which files are being downloaded. When complete, you'll see a summary of what was downloaded.

---

## Step 4: Generate the Terrain Model

Now you're ready to generate your 3D terrain model! Run this command:

```bash
bash <(curl -fsSL https://raw.githubusercontent.com/rdudhagra/topo-tessellate/main/run.sh) \
    --config configs/my_terrain.yaml \
    --topo-dir ./topo \
    --output-dir ./outputs
```

**What this does:**
- Downloads and runs the terrain generator inside a Docker container
- Reads your config file and processes the elevation data
- Creates 3D-printable STL files in your output folder

**This may take several minutes** depending on the size and complexity of your area.

### Understanding the Output

When complete, check your `outputs` folder. You'll find STL files that can be opened in 3D printing software:

| File | Description |
|------|-------------|
| `*_land.stl` | The terrain surface (mountains, valleys, etc.) |
| `*_base.stl` | The base/platform with interlocking joints |
| `*_buildings.stl` | 3D building footprints (if enabled) |

---

## Troubleshooting

### "Docker is not running"

Make sure Docker Desktop (Mac) or the Docker service (Linux) is running. On Mac, look for the Docker whale icon in your menu bar.

### "Permission denied"

On Linux, you may need to add `sudo` before docker commands, or add your user to the docker group:
```bash
sudo usermod -aG docker $USER
```
Then log out and log back in.

### "No elevation data found"

Make sure the coordinates in your config file are correct. The USGS data primarily covers the United States. For other countries, you may need to use SRTM data instead.

### The model is flat or has no detail

Try reducing the `downsample_factor` in your config (lower = more detail, but larger files and longer processing time).

---

## Tips for Best Results

1. **Start small**: Begin with a small area (1-2 km²) to test your settings before processing larger regions.

2. **Coordinate order matters**: Longitude comes before latitude. For the US, longitude is negative (west of Prime Meridian).

3. **Elevation exaggeration**: Real terrain often looks flat when 3D printed. Try `elevation_multiplier: 2` or higher to make features more visible.

4. **Tiling for large prints**: If your model exceeds your printer's build volume, enable tiling to split it into multiple interlocking pieces.