#!/usr/bin/env python3

import argparse
from glob import glob
import mimetypes
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import sys
import threading
from typing import Any, List, Optional
from types import ModuleType
from roop.processors.frame import face_swapper

import cv2
import insightface
import onnxruntime


from roop import core
from roop.face_analyser import get_one_face
import roop.globals
import tensorflow

TEMP_DIRECTORY = 'temp'
TEMP_VIDEO_FILE = 'temp.mp4'

FACE_SWAPPER = None
THREAD_LOCK = threading.Lock()

FRAME_PROCESSOR_MODULES: List[ModuleType] = []
FRAME_PROCESSORS_INTERFACE = [
    'pre_check',
    'pre_start',
    'process_frame',
    'process_frames',
    'process_image',
    'process_video',
    'post_process'
]


def get_face_swapper() -> Any:
    global FACE_SWAPPER

    with THREAD_LOCK:
        if FACE_SWAPPER is None:
            model_path = resolve_relative_path('../models/inswapper_128.onnx')
            FACE_SWAPPER = insightface.model_zoo.get_model(model_path, providers=roop.globals.execution_providers)
    return FACE_SWAPPER


def resolve_relative_path(path: str) -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), path))


def get_temp_directory_path(target_path: str) -> str:
    target_name, _ = os.path.splitext(os.path.basename(target_path))
    target_directory_path = os.path.dirname(target_path)
    return os.path.join(target_directory_path, TEMP_DIRECTORY, target_name)


def get_temp_output_path(target_path: str) -> str:
    temp_directory_path = get_temp_directory_path(target_path)
    return os.path.join(temp_directory_path, TEMP_VIDEO_FILE)

def create_temp(target_path: str) -> None:
    temp_directory_path = get_temp_directory_path(target_path)
    Path(temp_directory_path).mkdir(parents=True, exist_ok=True)


def move_temp(target_path: str, output_path: str) -> None:
    temp_output_path = get_temp_output_path(target_path)
    if os.path.isfile(temp_output_path):
        if os.path.isfile(output_path):
            os.remove(output_path)
        shutil.move(temp_output_path, output_path)


def clean_temp(target_path: str) -> None:
    temp_directory_path = get_temp_directory_path(target_path)
    parent_directory_path = os.path.dirname(temp_directory_path)
    if not roop.globals.keep_frames and os.path.isdir(temp_directory_path):
        shutil.rmtree(temp_directory_path)
    if os.path.exists(parent_directory_path) and not os.listdir(parent_directory_path):
        os.rmdir(parent_directory_path)


def destroy() -> None:
    if roop.globals.target_path:
        clean_temp(roop.globals.target_path)
    sys.exit()

def encode_execution_providers(execution_providers: List[str]) -> List[str]:
    return [execution_provider.replace('ExecutionProvider', '').lower() for execution_provider in execution_providers]


def decode_execution_providers(execution_providers: List[str]) -> List[str]:
    return [provider for provider, encoded_execution_provider in zip(onnxruntime.get_available_providers(), encode_execution_providers(onnxruntime.get_available_providers()))
            if any(execution_provider in encoded_execution_provider for execution_provider in execution_providers)]


def normalize_output_path(source_path: str, target_path: str, output_path: str) -> Optional[str]:
    if source_path and target_path and output_path:
        source_name, _ = os.path.splitext(os.path.basename(source_path))
        target_name, target_extension = os.path.splitext(os.path.basename(target_path))
        if os.path.isdir(output_path):
            return os.path.join(output_path, source_name + '-' + target_name + target_extension)
    return output_path

def suggest_execution_providers() -> List[str]:
    return encode_execution_providers(onnxruntime.get_available_providers())


def suggest_execution_threads() -> int:
    if 'CUDAExecutionProvider' in onnxruntime.get_available_providers():
        return 8
    return 1

def has_image_extension(image_path: str) -> bool:
    return image_path.lower().endswith(('png', 'jpg', 'jpeg', 'webp'))

def parse_args3() -> None:
    signal.signal(signal.SIGINT, lambda signal_number, frame: destroy())
    program = argparse.ArgumentParser(formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=100))
    program.add_argument('--alt-sources', help='single image, csv of images, or glob', required=True, dest='alt_source_paths')
    program.add_argument('--source', help='default source image', required=True, dest='source_path')

    program.add_argument('-t', '--target', help='select an target image or video', required=True, dest='target_path')
    program.add_argument('-o', '--output', help='select output file or directory', required=True, dest='output_path')
    program.add_argument('-f', '--output-format', help="output format (choices: (image, video)", dest="output_format", default='video', choices=['video', 'image'])
    program.add_argument('-q', '--quiet', help="quiet - reduce stdout output", dest='quiet', action='store_true')
    # program.add_argument('--frame-processor', help='frame processors (choices: face_swapper, face_enhancer, ...)', dest='frame_processor', default=['face_swapper'], nargs='+')
    # program.add_argument('--keep-fps', help='keep target fps', dest='keep_fps', action='store_true')
    program.add_argument('--keep-frames', help='keep temporary frames', dest='keep_frames', action='store_true')
    program.add_argument('--skip-audio', help='skip target audio', dest='skip_audio', action='store_true')
    program.add_argument('--many-faces', help='process every face', dest='many_faces', action='store_true')
    # program.add_argument('--reference-face-position', help='position of the reference face', dest='reference_face_position', type=int, default=0)
    # program.add_argument('--reference-frame-number', help='number of the reference frame', dest='reference_frame_number', type=int, default=0)
    program.add_argument('--similar-face-distance', help='face distance used for recognition', dest='similar_face_distance', type=float, default=0.85)
    program.add_argument('--temp-frame-format', help='image format used for frame extraction', dest='temp_frame_format', default='png', choices=['jpg', 'png'])
    program.add_argument('--temp-frame-quality', help='image quality used for frame extraction', dest='temp_frame_quality', type=int, default=0, choices=range(101), metavar='[0-100]')
    program.add_argument('--output-video-encoder', help='encoder used for the output video', dest='output_video_encoder', default='libx264', choices=['libx264', 'libx265', 'libvpx-vp9', 'h264_nvenc', 'hevc_nvenc'])
    program.add_argument('--output-video-quality', help='quality used for the output video', dest='output_video_quality', type=int, default=35, choices=range(101), metavar='[0-100]')
    program.add_argument('--max-memory', help='maximum amount of RAM in GB', dest='max_memory', type=int)
    program.add_argument('--execution-provider', help='available execution provider (choices: cpu, ...)', dest='execution_provider', default=['cpu'], choices=suggest_execution_providers(), nargs='+')
    program.add_argument('--execution-threads', help='number of execution threads', dest='execution_threads', type=int, default=suggest_execution_threads())
    program.add_argument('-v', '--version', action='version', version=f'{roop.metadata.name} {roop.metadata.version}')

    args = program.parse_args()
    return args


def expand_paths(val):
    paths = None
    if "," in val:
        print("csv")
        paths = [os.path.expanduser(p) for p in val.split(",")]
    elif "*" in val:
        print("glob")
        if "~" in val:
            val = os.path.expanduser(val)
        paths = list(glob(val))

    else:
        if "~" in val:
            val = os.path.expanduser(val)

        if os.path.isdir(val):
            print("dir")
            paths = []
            for nm in os.listdir(val):
                # print(nm)
                if os.path.isfile(f"{val}/{nm}"):
                    # print("is file")
                    ext = os.path.splitext(nm)[1]
                    # print(ext)
                    if ext in [".png", ".jpg", ".jpeg"]:
                        paths.append(f"{val}/{nm}")
        elif os.path.isfile(val):
            print("file")
            ext = os.path.splitext(val)[1]
            if ext in [".png", ".jpg", ".jpeg"]:
                paths = [val]
        else:
            print("WTF!")
            sys.exit(1)
    return paths


def set_globals(args):
    source_path = expand_paths(args.source_path)[0]
    print(f"source_path: {source_path}")
    roop.globals.source_path = source_path

    # handle csv, globs, homedirs, etc
    alt_source_paths = expand_paths(args.alt_source_paths)
    # remove source_path from alt_source_paths, if present
    alt_source_paths = [p for p in alt_source_paths if p != source_path]
    print(f"alt_source_paths: {alt_source_paths}")
    roop.globals.alt_source_paths = alt_source_paths

    tp = expand_paths(args.target_path)[0]
    print(f"target_path: {tp}")
    roop.globals.target_path = tp

    roop.globals.output_path = normalize_output_path(roop.globals.source_path, roop.globals.target_path, args.output_path)
    roop.globals.output_format = args.output_format
    roop.globals.quiet = args.quiet
    roop.globals.headless = True
    roop.globals.frame_processors = "face_swapper"
    roop.globals.keep_fps = True
    roop.globals.keep_frames = args.keep_frames
    roop.globals.skip_audio = args.skip_audio
    roop.globals.many_faces = args.many_faces
    roop.globals.reference_face_position = 0 #args.reference_face_position
    roop.globals.reference_frame_number = 0 #args.reference_frame_number
    roop.globals.similar_face_distance = args.similar_face_distance
    roop.globals.temp_frame_format = args.temp_frame_format
    roop.globals.temp_frame_quality = args.temp_frame_quality
    roop.globals.output_video_encoder = args.output_video_encoder
    roop.globals.output_video_quality = args.output_video_quality
    roop.globals.max_memory = args.max_memory
    roop.globals.execution_providers = decode_execution_providers(args.execution_provider)
    roop.globals.execution_threads = args.execution_threads


def limit_resources() -> None:
    # prevent tensorflow memory leak
    gpus = tensorflow.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tensorflow.config.experimental.set_virtual_device_configuration(gpu, [
            tensorflow.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)
        ])
    # limit memory usage
    if roop.globals.max_memory:
        memory = roop.globals.max_memory * 1024 ** 3
        if platform.system().lower() == 'darwin':
            memory = roop.globals.max_memory * 1024 ** 6
        if platform.system().lower() == 'windows':
            import ctypes
            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            kernel32.SetProcessWorkingSetSize(-1, ctypes.c_size_t(memory), ctypes.c_size_t(memory))
        else:
            import resource
            resource.setrlimit(resource.RLIMIT_DATA, (memory, memory))



def update_status(message: str, scope: str = 'ROOP.CORE') -> None:
    print(f'[{scope}] {message}')


def pre_check() -> bool:
    if sys.version_info < (3, 9):
        update_status('Python version is not supported - please upgrade to 3.9 or higher.')
        return False
    if not shutil.which('ffmpeg'):
        update_status('ffmpeg is not installed.')
        return False
    if not shutil.which('convert'):
        update_status('imagemagick is not installed')
        return False
    return True


def is_image(image_path: str) -> bool:
    if image_path and os.path.isfile(image_path):
        mimetype, _ = mimetypes.guess_type(image_path)
        return bool(mimetype and mimetype.startswith('image/'))
    return False


def check_for_face(image_path):
    print(f"checking for face in {image_path}")
    found = True
    img = cv2.imread(image_path)
    face = get_one_face(img)
    if not face or face is None:
        print(f"No face detected in {image_path}")
        found = False
    else:
        print("bbox", face['bbox'])
        print("kps", face["kps"])
        print("det_score", face["det_score"])
        print("landmark_3d_68", face["landmark_3d_68"].shape)
        print("pose", face["pose"].shape)
        print("landmark_2d_106", face["landmark_2d_106"].shape)
        print("gender", face.sex)
        print("age", face["age"])
        print("embedding", face["embedding"].shape)
        sys.exit(0)
    return found


def filter_out_source_images_without_faces() -> None:
    sources = []
    for source in roop.globals.alt_source_paths:
        # print(f"checking for face in {source}")
        if not is_image(source):
            print(f"{source} not an image. skipping")
            continue
        if check_for_face(source):
            # print("found")
            sources.append(source)
        else:
            print(f"no face found in {source}")
    roop.globals.source_paths = sources


def create_temp(target_path: str) -> None:
    temp_directory_path = get_temp_directory_path(target_path)
    Path(temp_directory_path).mkdir(parents=True, exist_ok=True)


def swap_image_to_image():
    # sinces it's a single frame cp target to output and then we'll
    # just swap src on top of it
    # not sure if it was meta-data, symlink, or google-drive, but copy2() doens't
    # work yet copy() does
    # shutil.copy2(roop.globals.target_path, roop.globals.output_path)
    shutil.copy(roop.globals.target_path, roop.globals.output_path)
    # process frame
    face_swapper.process_image(roop.globals.source_path, roop.globals.output_path, roop.globals.output_path)
    face_swapper.post_process()
    # validate image
    # update_status("validating output")
    if is_image(roop.globals.output_path):
        update_status('Processing to image succeed!')
        if not check_for_face(roop.globals.output_path):
            print(f"although no face found in {roop.globals.output_path}")
    else:
        update_status('Processing to image failed!')


def get_temp_frame_paths(target_path: str) -> List[str]:
    temp_directory_path = get_temp_directory_path(target_path)
    return glob.glob((os.path.join(glob.escape(temp_directory_path), '*.' + roop.globals.temp_frame_format)))


def get_temp_directory_path(target_path: str) -> str:
    target_name, _ = os.path.splitext(os.path.basename(target_path))
    target_directory_path = os.path.dirname(target_path)
    return os.path.join(target_directory_path, TEMP_DIRECTORY, target_name)


def get_temp_output_path(target_path: str) -> str:
    temp_directory_path = get_temp_directory_path(target_path)
    return os.path.join(temp_directory_path, TEMP_VIDEO_FILE)

def swap_image_to_target_frames():
    # create tempdir and extract frames to it
    create_temp(roop.globals.target_path)
    # extract frames
    if roop.globals.keep_fps:
        fps = detect_fps(roop.globals.target_path)
        # update_status(f'Extracting frames with {fps} FPS...')
        extract_frames(roop.globals.target_path, fps)
    else:
        # update_status('Extracting frames with 30 FPS...')
        extract_frames(roop.globals.target_path)

    temp_frame_paths = get_temp_frame_paths(roop.globals.target_path)


def run_ffmpeg(args: List[str]) -> bool:
    commands = ['ffmpeg', '-hide_banner', '-loglevel', roop.globals.log_level]
    commands.extend(args)
    # print(" ".join(commands))
    try:
        subprocess.check_output(commands, stderr=subprocess.STDOUT)
        return True
    except Exception:
        pass
    return False



def run_imagemagick(args: List[str]) -> bool:
    commands = ['convert']
    commands.extend(args)
    # print(" ".join(commands))
    try:
        subprocess.check_output(commands, stderr=subprocess.STDOUT)
        return True
    except Exception:
        pass
    return False

def extract_frames(target_path: str, fps: float = 30) -> bool:
    temp_directory_path = get_temp_directory_path(target_path)
    temp_frame_quality = roop.globals.temp_frame_quality * 31 // 100
    return run_ffmpeg(['-hwaccel', 'auto', '-i', target_path, '-q:v', str(temp_frame_quality), '-pix_fmt', 'rgb24', '-vf', 'fps=' + str(fps), os.path.join(temp_directory_path, '%04d.' + roop.globals.temp_frame_format)])


def detect_fps(target_path: str) -> float:
    command = ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=r_frame_rate', '-of', 'default=noprint_wrappers=1:nokey=1', target_path]
    output = subprocess.check_output(command).decode().strip().split('/')
    try:
        numerator, denominator = map(int, output)
        return numerator / denominator
    except Exception:
        pass
    return 30


def create_video(target_path: str, fps: float = 30) -> bool:
    temp_output_path = get_temp_output_path(target_path)
    temp_directory_path = get_temp_directory_path(target_path)
    output_video_quality = (roop.globals.output_video_quality + 1) * 51 // 100
    commands = ['-hwaccel', 'auto', '-r', str(fps), '-i', os.path.join(temp_directory_path, '%04d.' + roop.globals.temp_frame_format), '-c:v', roop.globals.output_video_encoder]
    if roop.globals.output_video_encoder in ['libx264', 'libx265', 'libvpx']:
        commands.extend(['-crf', str(output_video_quality)])
    if roop.globals.output_video_encoder in ['h264_nvenc', 'hevc_nvenc']:
        commands.extend(['-cq', str(output_video_quality)])
    commands.extend(['-pix_fmt', 'yuv420p', '-vf', 'colorspace=bt709:iall=bt601-6-625:fast=1', '-y', temp_output_path])

    return run_ffmpeg(commands)


def create_gif(target_path: str, fps: float = 30) -> bool:
    delay = (1.0 / fps) * 100
    temp_output_path = get_temp_output_path(target_path)
    temp_directory_path = get_temp_directory_path(target_path)
    commands = ['-delay', str(delay), '-loop', '0', os.path.join(temp_directory_path, '*.' + roop.globals.temp_frame_format), roop.globals.output_path]

    return run_imagemagick(commands)


def merge_frames_to_gif():
    fps = detect_fps(roop.globals.target_path)
    create_gif(roop.globals.target_path, fps)
    roop.globals.keep_frames = False



def merge_frames_to_video():
  fps = detect_fps(roop.globals.target_path)
  if roop.globals.keep_fps:
    #   fps = detect_fps(roop.globals.target_path)
      # update_status(f'Creating video with {fps} FPS...')
      create_video(roop.globals.target_path, fps)
  else:
      # update_status('Creating video with 30 FPS...')
      create_video(roop.globals.target_path)

  # handle audio
  if roop.globals.skip_audio:
      move_temp(roop.globals.target_path, roop.globals.output_path)
      update_status('Skipping audio...')
  else:
      if roop.globals.keep_fps:
          update_status('Restoring audio...')
          restore_audio(roop.globals.target_path, roop.globals.output_path)
      elif fps != float(30):
          update_status('Restoring audio might cause issues as fps are not kept...')
          restore_audio(roop.globals.target_path, roop.globals.output_path)


def restore_audio(target_path: str, output_path: str) -> None:
    temp_output_path = get_temp_output_path(target_path)
    done = run_ffmpeg(['-i', temp_output_path, '-i', target_path, '-c:v', 'copy', '-map', '0:v:0', '-map', '1:a:0', '-y', output_path])
    if not done:
        move_temp(target_path, output_path)


def start() -> None:
    filter_out_source_images_without_faces()
    # if target is a static image, verify it has a face in it as well
    if has_image_extension(roop.globals.target_path):
        if not check_for_face(roop.globals.target_path):
            print(f"no face found in {roop.globals.target_path}")
            sys.exit(1)
        swap_image_to_image()
    else:
        swap_image_to_target_frames()
        # reconstruct either video or gif from temp frames
        print("output_path", roop.globals.output_path)
        if roop.globals.output_path.endswith(".gif"):
            merge_frames_to_gif()
        else:
            merge_frames_to_video()


def main() -> None:
    args = parse_args3()
    set_globals(args)
    if not pre_check():
        print("pre_check() failed")
        sys.exit(1)
    if not face_swapper.pre_check():
        print("face_swapper.pre_check() failed")
        sys.exit(1)
    start()


if __name__ == '__main__':
    main()
