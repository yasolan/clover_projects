#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import math
import threading

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from flask import Flask, request, jsonify, render_template_string, Response
from sensor_msgs.msg import Image, CompressedImage
from std_srvs.srv import Trigger as TriggerSrv
from clover import srv as clover_srv


app = Flask(__name__)

MAP_PATH = '/home/pi/catkin_ws/src/clover/aruco_pose/map/test_map.txt'
FRAME_ID = 'aruco_map'

bridge = CvBridge()

navigate = None
get_telemetry = None
land = None

ros_ready = False

camera_lock = threading.Lock()
latest_frame_jpeg = None
camera_subscriber = None
current_camera_topic = None
current_camera_type = None

mission_lock = threading.Lock()
mission_thread = None
mission_stop_event = threading.Event()

mission_status = {
    'running': False,
    'message': 'Маршрут не запущен',
    'current_index': -1,
    'route': [],
    'height': 1.5,
    'speed': 0.5,
    'home_point': None,
    'final_point': None,
    'land_at_end': False
}

HTML_PAGE = """
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <title>Маршрут по ArUco-карте</title>
    <style>
        body {
            margin: 0;
            font-family: Arial, sans-serif;
            background: #0f172a;
            color: #e2e8f0;
        }

        .layout {
            display: flex;
            min-height: 100vh;
        }

        .left {
            flex: 1;
            padding: 20px;
            box-sizing: border-box;
        }

        .right {
            width: 420px;
            padding: 20px;
            box-sizing: border-box;
            background: #111827;
            border-left: 1px solid rgba(255,255,255,0.08);
        }

        h1 {
            margin-top: 0;
            font-size: 28px;
        }

        .status-bar {
            margin-bottom: 15px;
            padding: 12px 14px;
            border-radius: 10px;
            background: rgba(255,255,255,0.05);
            line-height: 1.5;
        }

        .panel {
            background: rgba(255,255,255,0.05);
            border-radius: 14px;
            padding: 16px;
            margin-bottom: 16px;
        }

        .panel h3 {
            margin-top: 0;
            font-size: 20px;
        }

        .controls-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 12px;
        }

        label {
            display: block;
            font-size: 14px;
            margin-bottom: 6px;
            color: #cbd5e1;
        }

        input, select, textarea, button {
            width: 100%;
            box-sizing: border-box;
            border-radius: 10px;
            border: none;
            padding: 10px 12px;
            font-size: 15px;
        }

        input[type="checkbox"] {
            width: auto;
            transform: scale(1.2);
            margin-right: 8px;
            vertical-align: middle;
        }

        .checkbox-row {
            display: flex;
            align-items: center;
            padding: 11px 12px;
            background: rgba(255,255,255,0.06);
            border-radius: 10px;
            min-height: 42px;
        }

        .checkbox-row label {
            margin: 0;
            cursor: pointer;
            color: #e2e8f0;
        }

        button {
            cursor: pointer;
            font-weight: bold;
        }

        .btn-green { background: #22c55e; color: white; }
        .btn-red { background: #ef4444; color: white; }
        .btn-yellow { background: #f59e0b; color: white; }
        .btn-blue { background: #3b82f6; color: white; }
        .btn-gray { background: #475569; color: white; }
        .btn-darkred { background: #b91c1c; color: white; }

        .buttons-2 {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 12px;
        }

        .buttons-3 {
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            gap: 12px;
        }

        .map-wrap {
            position: relative;
            background: #020617;
            border-radius: 14px;
            padding: 10px;
            overflow: hidden;
        }

        #mapCanvas {
            width: 100%;
            height: 700px;
            display: block;
            background: #0b1220;
            border-radius: 10px;
            cursor: crosshair;
        }

        .legend {
            margin-top: 10px;
            color: #cbd5e1;
            font-size: 14px;
            line-height: 1.45;
        }

        .camera-box {
            position: sticky;
            top: 20px;
        }

        .camera-frame {
            width: 100%;
            background: black;
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid rgba(255,255,255,0.1);
        }

        .camera-frame img {
            width: 100%;
            display: block;
        }

        .route-list {
            font-size: 14px;
            line-height: 1.5;
            color: #cbd5e1;
            white-space: pre-wrap;
        }

        .small {
            font-size: 13px;
            color: #94a3b8;
        }

        @media (max-width: 1200px) {
            .layout {
                flex-direction: column;
            }

            .right {
                width: 100%;
                border-left: none;
                border-top: 1px solid rgba(255,255,255,0.08);
            }

            #mapCanvas {
                height: 520px;
            }
        }
    </style>
</head>
<body>
<div class="layout">
    <div class="left">
        <h1>Полет по ArUco-карте</h1>

        <div class="status-bar">
            ROS: <span id="rosStatus">...</span><br>
            Маршрут: <span id="missionStatus">...</span><br>
            Текущая точка: <span id="missionIndex">-</span><br>
            Домашняя точка: <span id="homePoint">-</span><br>
            Финальная точка: <span id="finalPoint">-</span><br>
            Посадка в конце: <span id="landAtEndStatus">нет</span>
        </div>

        <div class="panel">
            <h3>Управление маршрутом</h3>

            <div class="controls-grid">
                <div>
                    <label for="heightInput">Высота, м</label>
                    <input type="number" id="heightInput" value="1.5" step="0.1" min="0.3">
                </div>
                <div>
                    <label for="speedInput">Скорость, м/с</label>
                    <input type="number" id="speedInput" value="0.5" step="0.1" min="0.1">
                </div>
                <div>
                    <label for="routeMode">Режим задания маршрута</label>
                    <select id="routeMode">
                        <option value="click">Кликами по меткам</option>
                        <option value="manual">Вручную списком ID</option>
                    </select>
                </div>
                <div>
                    <label for="manualRoute">Маршрут ID через запятую</label>
                    <input type="text" id="manualRoute" placeholder="Например: 0,3,7,2">
                </div>
                <div style="grid-column: 1 / span 2;">
                    <div class="checkbox-row">
                        <input type="checkbox" id="landAtEnd">
                        <label for="landAtEnd">Садиться в конце маршрута</label>
                    </div>
                </div>
            </div>

            <div style="margin-top: 12px;" class="buttons-3">
                <button class="btn-green" onclick="startMission()">Старт</button>
                <button class="btn-yellow" onclick="stopMission()">Стоп</button>
                <button class="btn-red" onclick="landDrone()">Посадка</button>
            </div>

            <div style="margin-top: 12px;" class="buttons-2">
                <button class="btn-blue" onclick="removeLastPoint()">Удалить последнюю точку</button>
                <button class="btn-gray" onclick="clearRoute()">Очистить маршрут</button>
            </div>
        </div>

        <div class="panel">
            <h3>Аварийные действия</h3>

            <div class="controls-grid">
                <div>
                    <label for="emergencyMode">Что делать при аварии</label>
                    <select id="emergencyMode">
                        <option value="land_now">Сразу сесть</option>
                        <option value="go_final_land">Уйти в финальную точку и сесть</option>
                        <option value="go_home_land">Вернуться в домашнюю точку и сесть</option>
                    </select>
                </div>
                <div>
                    <label>&nbsp;</label>
                    <button class="btn-darkred" onclick="emergencyAction()">АВАРИЙНОЕ ДЕЙСТВИЕ</button>
                </div>
            </div>

            <div class="legend">
                Домашняя точка запоминается в момент запуска маршрута. Финальная точка — это последняя точка текущего маршрута.
            </div>
        </div>

        <div class="panel">
            <h3>Карта ArUco</h3>
            <div class="map-wrap">
                <canvas id="mapCanvas"></canvas>
            </div>
            <div class="legend">
                ЛКМ по метке — добавить ее в маршрут. Если метка выбрана несколько раз, на ней будут показаны все номера проходов.
            </div>
        </div>

        <div class="panel">
            <h3>Текущий маршрут</h3>
            <div class="route-list" id="routeList">Пока пусто</div>
        </div>
    </div>

    <div class="right">
        <div class="camera-box">
            <div class="panel">
                <h3>Камера</h3>
                <label for="cameraTopicSelect">Топик камеры</label>
                <select id="cameraTopicSelect"></select>

                <div style="margin-top: 10px;" class="buttons-2">
                    <button class="btn-blue" onclick="applyCameraTopic()">Подключить топик</button>
                    <button class="btn-gray" onclick="loadTopics()">Обновить список</button>
                </div>

                <div class="small" style="margin-top: 10px;">
                    Лучше выбирать compressed-топики, если они есть. Они работают быстрее.
                </div>
            </div>

            <div class="camera-frame">
                <img id="cameraFeed" src="/camera_feed" alt="camera">
            </div>
        </div>
    </div>
</div>

<script>
    const canvas = document.getElementById('mapCanvas');
    const ctx = canvas.getContext('2d');

    let markers = [];
    let route = [];
    let bounds = null;
    let hoveredMarkerId = null;
    let activeMissionIndex = -1;

    function resizeCanvas() {
        const rect = canvas.getBoundingClientRect();
        canvas.width = rect.width;
        canvas.height = rect.height;
        drawMap();
    }

    window.addEventListener('resize', resizeCanvas);

    function worldToCanvas(x, y) {
        if (!bounds) return {x: 0, y: 0};

        const pad = 50;
        const width = canvas.width - pad * 2;
        const height = canvas.height - pad * 2;

        const dx = Math.max(bounds.max_x - bounds.min_x, 0.001);
        const dy = Math.max(bounds.max_y - bounds.min_y, 0.001);
        const scale = Math.min(width / dx, height / dy);

        const px = pad + (x - bounds.min_x) * scale;
        const py = canvas.height - pad - (y - bounds.min_y) * scale;

        return {x: px, y: py};
    }

    function drawGrid() {
        ctx.strokeStyle = 'rgba(255,255,255,0.06)';
        ctx.lineWidth = 1;

        for (let i = 0; i < canvas.width; i += 50) {
            ctx.beginPath();
            ctx.moveTo(i, 0);
            ctx.lineTo(i, canvas.height);
            ctx.stroke();
        }

        for (let i = 0; i < canvas.height; i += 50) {
            ctx.beginPath();
            ctx.moveTo(0, i);
            ctx.lineTo(canvas.width, i);
            ctx.stroke();
        }
    }

    function getRouteIndicesForMarker(markerId) {
        const result = [];
        for (let i = 0; i < route.length; i++) {
            if (route[i] === markerId) {
                result.push(i + 1);
            }
        }
        return result;
    }

    function drawRouteBadges(baseX, baseY, indices, isActiveMarker) {
        if (!indices.length) return;

        for (let i = 0; i < indices.length; i++) {
            const bx = baseX + 22;
            const by = baseY - 18 + i * 18;

            ctx.fillStyle = isActiveMarker && indices[i] === activeMissionIndex + 1 ? '#ef4444' : '#020617';
            ctx.fillRect(bx, by, 24, 16);

            ctx.strokeStyle = 'white';
            ctx.lineWidth = 1.5;
            ctx.strokeRect(bx, by, 24, 16);

            ctx.fillStyle = 'white';
            ctx.font = '11px Arial';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(String(indices[i]), bx + 12, by + 8);
        }
    }

    function drawMap() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        drawGrid();

        if (!markers.length) {
            ctx.fillStyle = '#e2e8f0';
            ctx.font = '20px Arial';
            ctx.fillText('Карта не загружена', 30, 40);
            return;
        }

        if (route.length > 1) {
            ctx.strokeStyle = '#22c55e';
            ctx.lineWidth = 4;
            ctx.beginPath();

            for (let i = 0; i < route.length; i++) {
                const marker = markers.find(m => m.id === route[i]);
                if (!marker) continue;

                const p = worldToCanvas(marker.x, marker.y);
                if (i === 0) ctx.moveTo(p.x, p.y);
                else ctx.lineTo(p.x, p.y);
            }

            ctx.stroke();
        }

        for (const marker of markers) {
            const p = worldToCanvas(marker.x, marker.y);
            const routeIndices = getRouteIndicesForMarker(marker.id);
            const inRoute = routeIndices.length > 0;
            const isActiveMarker = routeIndices.includes(activeMissionIndex + 1);

            ctx.beginPath();
            ctx.arc(p.x, p.y, inRoute ? 18 : 14, 0, Math.PI * 2);

            if (isActiveMarker) {
                ctx.fillStyle = '#ef4444';
            } else if (hoveredMarkerId === marker.id) {
                ctx.fillStyle = '#f59e0b';
            } else if (inRoute) {
                ctx.fillStyle = '#22c55e';
            } else {
                ctx.fillStyle = '#3b82f6';
            }

            ctx.fill();
            ctx.strokeStyle = 'white';
            ctx.lineWidth = 2;
            ctx.stroke();

            ctx.fillStyle = 'white';
            ctx.font = 'bold 14px Arial';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(marker.id, p.x, p.y);

            ctx.fillStyle = '#cbd5e1';
            ctx.font = '12px Arial';
            ctx.fillText(`(${marker.x.toFixed(2)}, ${marker.y.toFixed(2)})`, p.x, p.y + 26);

            drawRouteBadges(p.x, p.y, routeIndices, isActiveMarker);
        }
    }

    function updateRouteList() {
        const routeList = document.getElementById('routeList');

        if (!route.length) {
            routeList.textContent = 'Пока пусто';
            return;
        }

        let text = '';
        for (let i = 0; i < route.length; i++) {
            const marker = markers.find(m => m.id === route[i]);
            if (!marker) continue;
            text += `${i + 1}. ID ${marker.id} -> x=${marker.x.toFixed(2)}, y=${marker.y.toFixed(2)}, z=${marker.z.toFixed(2)}\\n`;
        }

        routeList.textContent = text;
    }

    function computeBounds(data) {
        if (!data.length) return null;

        let min_x = data[0].x;
        let max_x = data[0].x;
        let min_y = data[0].y;
        let max_y = data[0].y;

        for (const m of data) {
            min_x = Math.min(min_x, m.x);
            max_x = Math.max(max_x, m.x);
            min_y = Math.min(min_y, m.y);
            max_y = Math.max(max_y, m.y);
        }

        const margin = 0.5;
        return {
            min_x: min_x - margin,
            max_x: max_x + margin,
            min_y: min_y - margin,
            max_y: max_y + margin
        };
    }

    function getMarkerAt(mx, my) {
        for (const marker of markers) {
            const p = worldToCanvas(marker.x, marker.y);
            const d = Math.sqrt((mx - p.x) ** 2 + (my - p.y) ** 2);
            if (d <= 20) return marker;
        }
        return null;
    }

    canvas.addEventListener('mousemove', function(e) {
        const rect = canvas.getBoundingClientRect();
        const mx = e.clientX - rect.left;
        const my = e.clientY - rect.top;
        const marker = getMarkerAt(mx, my);
        hoveredMarkerId = marker ? marker.id : null;
        drawMap();
    });

    canvas.addEventListener('click', function(e) {
        const routeMode = document.getElementById('routeMode').value;
        if (routeMode !== 'click') return;

        const rect = canvas.getBoundingClientRect();
        const mx = e.clientX - rect.left;
        const my = e.clientY - rect.top;
        const marker = getMarkerAt(mx, my);

        if (marker) {
            route.push(marker.id);
            drawMap();
            updateRouteList();
        }
    });

    async function loadMap() {
        const response = await fetch('/api/map');
        const data = await response.json();

        if (!data.success) {
            alert(data.error || 'Не удалось загрузить карту');
            return;
        }

        markers = data.markers;
        bounds = computeBounds(markers);
        drawMap();
        updateRouteList();
    }

    async function loadTopics() {
        const response = await fetch('/api/topics');
        const data = await response.json();

        const select = document.getElementById('cameraTopicSelect');
        select.innerHTML = '';

        if (!data.success) {
            const opt = document.createElement('option');
            opt.textContent = 'Не удалось загрузить топики';
            select.appendChild(opt);
            return;
        }

        for (const topic of data.topics) {
            const opt = document.createElement('option');
            opt.value = topic;
            opt.textContent = topic;
            if (topic === data.current_camera_topic) {
                opt.selected = true;
            }
            select.appendChild(opt);
        }
    }

    async function applyCameraTopic() {
        const topic = document.getElementById('cameraTopicSelect').value;

        const response = await fetch('/api/camera_topic', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({topic: topic})
        });

        const data = await response.json();

        if (!data.success) {
            alert(data.error || 'Не удалось переключить топик');
            return;
        }

        const img = document.getElementById('cameraFeed');
        img.src = '/camera_feed?ts=' + Date.now();
    }

    function clearRoute() {
        route = [];
        drawMap();
        updateRouteList();
    }

    function removeLastPoint() {
        if (route.length > 0) {
            route.pop();
            drawMap();
            updateRouteList();
        }
    }

    async function startMission() {
        const routeMode = document.getElementById('routeMode').value;
        const height = parseFloat(document.getElementById('heightInput').value);
        const speed = parseFloat(document.getElementById('speedInput').value);
        const landAtEnd = document.getElementById('landAtEnd').checked;

        let routeIds = [];

        if (routeMode === 'click') {
            routeIds = route.slice();
        } else {
            const raw = document.getElementById('manualRoute').value.trim();
            if (raw.length) {
                routeIds = raw.split(',').map(x => parseInt(x.trim(), 10)).filter(x => !isNaN(x));
            }
        }

        const response = await fetch('/api/start_mission', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                route: routeIds,
                height: height,
                speed: speed,
                land_at_end: landAtEnd
            })
        });

        const data = await response.json();

        if (!data.success) {
            alert(data.error || 'Не удалось запустить маршрут');
            return;
        }

        route = routeIds.slice();
        drawMap();
        updateRouteList();
    }

    async function stopMission() {
        const response = await fetch('/api/stop_mission', {method: 'POST'});
        const data = await response.json();
        if (!data.success) {
            alert(data.error || 'Не удалось остановить маршрут');
        }
    }

    async function landDrone() {
        const response = await fetch('/api/land', {method: 'POST'});
        const data = await response.json();
        if (!data.success) {
            alert(data.error || 'Не удалось посадить дрон');
        }
    }

    async function emergencyAction() {
        const mode = document.getElementById('emergencyMode').value;

        const response = await fetch('/api/emergency', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({mode: mode})
        });

        const data = await response.json();
        if (!data.success) {
            alert(data.error || 'Не удалось выполнить аварийное действие');
        }
    }

    function formatPoint(point) {
        if (!point) return '-';
        return `x=${point.x.toFixed(2)}, y=${point.y.toFixed(2)}, z=${point.z.toFixed(2)}`;
    }

    async function updateStatus() {
        const response = await fetch('/api/status');
        const data = await response.json();

        document.getElementById('rosStatus').textContent = data.ros_ready ? 'готов' : 'не готов';
        document.getElementById('missionStatus').textContent = data.mission.message;
        document.getElementById('missionIndex').textContent = data.mission.current_index >= 0 ? (data.mission.current_index + 1) : '-';
        document.getElementById('homePoint').textContent = formatPoint(data.mission.home_point);
        document.getElementById('finalPoint').textContent = formatPoint(data.mission.final_point);
        document.getElementById('landAtEndStatus').textContent = data.mission.land_at_end ? 'да' : 'нет';

        activeMissionIndex = data.mission.current_index;
        drawMap();
    }

    async function reloadAll() {
        await loadMap();
        await loadTopics();
        await updateStatus();
    }

    setInterval(updateStatus, 700);

    resizeCanvas();
    reloadAll();
</script>
</body>
</html>
"""


def parse_aruco_map(map_path):
    if not os.path.exists(map_path):
        raise FileNotFoundError('Файл карты не найден: {}'.format(map_path))

    markers = []

    with open(map_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split()
            if len(parts) < 8:
                continue

            markers.append({
                'id': int(parts[0]),
                'length': float(parts[1]),
                'x': float(parts[2]),
                'y': float(parts[3]),
                'z': float(parts[4]),
                'rot_z': float(parts[5]),
                'rot_y': float(parts[6]),
                'rot_x': float(parts[7]),
            })

    markers.sort(key=lambda m: m['id'])
    return markers


def get_published_topics_with_types():
    result = {}
    try:
        for topic, msg_type in rospy.get_published_topics():
            result[topic] = msg_type
    except Exception:
        pass
    return result


def get_camera_topics():
    topics_with_types = get_published_topics_with_types()
    topics = []

    for topic, msg_type in topics_with_types.items():
        if msg_type in ('sensor_msgs/Image', 'sensor_msgs/CompressedImage'):
            if 'image' in topic.lower() or 'camera' in topic.lower():
                topics.append(topic)

    def topic_priority(name):
        score = 100
        low = name.lower()
        if 'compressed' in low:
            score -= 50
        if 'throttled' in low:
            score -= 20
        if 'main_camera' in low:
            score -= 10
        return score, name

    topics = sorted(list(set(topics)), key=topic_priority)
    return topics


def raw_image_callback(msg):
    global latest_frame_jpeg

    try:
        frame = bridge.imgmsg_to_cv2(msg, 'bgr8')

        h, w = frame.shape[:2]
        max_w = 640
        if w > max_w:
            new_h = int(h * max_w / float(w))
            frame = cv2.resize(frame, (max_w, new_h))

        ok, encoded = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])

        if ok:
            with camera_lock:
                latest_frame_jpeg = encoded.tobytes()
    except Exception:
        pass


def compressed_image_callback(msg):
    global latest_frame_jpeg

    try:
        with camera_lock:
            latest_frame_jpeg = bytes(msg.data)
    except Exception:
        pass


def subscribe_camera(topic_name):
    global camera_subscriber, current_camera_topic, current_camera_type, latest_frame_jpeg

    topics_with_types = get_published_topics_with_types()
    msg_type = topics_with_types.get(topic_name)

    if msg_type is None:
        raise RuntimeError('Топик {} не найден'.format(topic_name))

    if camera_subscriber is not None:
        try:
            camera_subscriber.unregister()
        except Exception:
            pass
        camera_subscriber = None

    latest_frame_jpeg = None
    current_camera_topic = topic_name
    current_camera_type = msg_type

    if msg_type == 'sensor_msgs/CompressedImage':
        camera_subscriber = rospy.Subscriber(
            topic_name,
            CompressedImage,
            compressed_image_callback,
            queue_size=1,
            tcp_nodelay=True
        )
    elif msg_type == 'sensor_msgs/Image':
        camera_subscriber = rospy.Subscriber(
            topic_name,
            Image,
            raw_image_callback,
            queue_size=1,
            buff_size=2**24,
            tcp_nodelay=True
        )
    else:
        raise RuntimeError('Неподдерживаемый тип топика: {}'.format(msg_type))


def set_mission_status(**kwargs):
    with mission_lock:
        mission_status.update(kwargs)


def navigate_wait(x=0.0, y=0.0, z=0.0, speed=0.5, frame_id='body', tolerance=0.18, auto_arm=False):
    res = navigate(
        x=x,
        y=y,
        z=z,
        yaw=float('nan'),
        speed=speed,
        frame_id=frame_id,
        auto_arm=auto_arm
    )

    if not res.success:
        raise RuntimeError(res.message)

    while not rospy.is_shutdown():
        if mission_stop_event.is_set():
            return False

        telem = get_telemetry(frame_id='navigate_target')
        dist = math.sqrt(telem.x ** 2 + telem.y ** 2 + telem.z ** 2)

        if dist < tolerance:
            return True

        time.sleep(0.12)

    return False


def get_current_point_in_map():
    telem = get_telemetry(frame_id=FRAME_ID)
    return {
        'x': float(telem.x),
        'y': float(telem.y),
        'z': float(telem.z)
    }


def mission_worker(route_ids, height, speed, land_at_end):
    try:
        markers = parse_aruco_map(MAP_PATH)
        marker_by_id = {m['id']: m for m in markers}

        home_point = get_current_point_in_map()
        final_marker = marker_by_id.get(route_ids[-1])

        final_point = None
        if final_marker is not None:
            final_point = {
                'x': final_marker['x'],
                'y': final_marker['y'],
                'z': height
            }

        set_mission_status(
            running=True,
            message='Маршрут выполняется',
            current_index=-1,
            route=route_ids[:],
            height=height,
            speed=speed,
            home_point=home_point,
            final_point=final_point,
            land_at_end=land_at_end
        )

        for index, marker_id in enumerate(route_ids):
            if mission_stop_event.is_set():
                set_mission_status(
                    running=False,
                    message='Маршрут остановлен',
                    current_index=-1
                )
                return

            if marker_id not in marker_by_id:
                raise RuntimeError('Метка ID {} отсутствует в карте'.format(marker_id))

            marker = marker_by_id[marker_id]

            set_mission_status(
                message='Полет к метке ID {}'.format(marker_id),
                current_index=index
            )

            reached = navigate_wait(
                x=marker['x'],
                y=marker['y'],
                z=height,
                speed=speed,
                frame_id=FRAME_ID,
                tolerance=0.18,
                auto_arm=(index == 0)
            )

            if not reached and mission_stop_event.is_set():
                set_mission_status(
                    running=False,
                    message='Маршрут остановлен',
                    current_index=-1
                )
                return

            time.sleep(0.25)

        if land_at_end:
            set_mission_status(
                message='Маршрут завершен, выполняется посадка',
                current_index=-1
            )

            res = land()
            if not res.success:
                raise RuntimeError(res.message)

            set_mission_status(
                running=False,
                message='Маршрут завершен, посадка выполнена',
                current_index=-1
            )
        else:
            set_mission_status(
                running=False,
                message='Маршрут завершен',
                current_index=-1
            )

    except Exception as e:
        set_mission_status(
            running=False,
            message='Ошибка маршрута: {}'.format(e),
            current_index=-1
        )


def emergency_worker(mode):
    try:
        with mission_lock:
            saved_final = mission_status.get('final_point')
            saved_home = mission_status.get('home_point')
            saved_height = mission_status.get('height', 1.5)
            saved_speed = mission_status.get('speed', 0.5)

        mission_stop_event.set()

        if mode == 'land_now':
            set_mission_status(
                running=False,
                message='Авария: немедленная посадка',
                current_index=-1
            )
            res = land()
            if not res.success:
                raise RuntimeError(res.message)
            return

        if mode == 'go_final_land':
            if not saved_final:
                raise RuntimeError('Финальная точка не задана')
            set_mission_status(
                running=False,
                message='Авария: полет в финальную точку и посадка',
                current_index=-1
            )
            navigate(
                x=saved_final['x'],
                y=saved_final['y'],
                z=max(saved_final['z'], 0.8),
                yaw=float('nan'),
                speed=max(saved_speed, 0.3),
                frame_id=FRAME_ID,
                auto_arm=False
            )
            while True:
                telem = get_telemetry(frame_id='navigate_target')
                dist = math.sqrt(telem.x ** 2 + telem.y ** 2 + telem.z ** 2)
                if dist < 0.2:
                    break
                time.sleep(0.12)

            res = land()
            if not res.success:
                raise RuntimeError(res.message)
            set_mission_status(message='Авария: финальная точка достигнута, посадка выполнена')
            return

        if mode == 'go_home_land':
            if not saved_home:
                raise RuntimeError('Домашняя точка не задана')
            set_mission_status(
                running=False,
                message='Авария: возврат домой и посадка',
                current_index=-1
            )
            navigate(
                x=saved_home['x'],
                y=saved_home['y'],
                z=max(saved_height, 0.8),
                yaw=float('nan'),
                speed=max(saved_speed, 0.3),
                frame_id=FRAME_ID,
                auto_arm=False
            )
            while True:
                telem = get_telemetry(frame_id='navigate_target')
                dist = math.sqrt(telem.x ** 2 + telem.y ** 2 + telem.z ** 2)
                if dist < 0.2:
                    break
                time.sleep(0.12)

            res = land()
            if not res.success:
                raise RuntimeError(res.message)
            set_mission_status(message='Авария: возврат домой выполнен, посадка выполнена')
            return

        raise RuntimeError('Неизвестный аварийный режим: {}'.format(mode))

    except Exception as e:
        set_mission_status(
            running=False,
            message='Ошибка аварийного действия: {}'.format(e),
            current_index=-1
        )


def init_ros():
    global navigate, get_telemetry, land, ros_ready

    rospy.init_node('aruco_web_route_panel', anonymous=True, disable_signals=True)

    try:
        rospy.wait_for_service('navigate', timeout=10)
        rospy.wait_for_service('get_telemetry', timeout=10)
        rospy.wait_for_service('land', timeout=10)

        navigate = rospy.ServiceProxy('navigate', clover_srv.Navigate)
        get_telemetry = rospy.ServiceProxy('get_telemetry', clover_srv.GetTelemetry)
        land = rospy.ServiceProxy('land', TriggerSrv)

        ros_ready = True

        topics = get_camera_topics()
        if topics:
            subscribe_camera(topics[0])

        rospy.loginfo('ROS и сервисы успешно инициализированы')
    except Exception as e:
        ros_ready = False
        rospy.logerr('Ошибка инициализации ROS: %s', e)


@app.route('/')
def index():
    return render_template_string(HTML_PAGE)


@app.route('/api/map')
def api_map():
    try:
        markers = parse_aruco_map(MAP_PATH)
        return jsonify({
            'success': True,
            'markers': markers,
            'map_path': MAP_PATH
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/topics')
def api_topics():
    try:
        return jsonify({
            'success': True,
            'topics': get_camera_topics(),
            'current_camera_topic': current_camera_topic,
            'current_camera_type': current_camera_type
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'topics': [],
            'current_camera_topic': current_camera_topic,
            'current_camera_type': current_camera_type
        }), 500


@app.route('/api/camera_topic', methods=['POST'])
def api_camera_topic():
    try:
        data = request.get_json(force=True)
        topic = data.get('topic', '').strip()

        if not topic:
            raise RuntimeError('Не указан топик камеры')

        subscribe_camera(topic)

        return jsonify({
            'success': True,
            'topic': current_camera_topic,
            'type': current_camera_type
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/camera_feed')
def camera_feed():
    def generate():
        while True:
            with camera_lock:
                frame = latest_frame_jpeg

            if frame is None:
                blank = np.zeros((240, 320, 3), dtype=np.uint8)
                cv2.putText(blank, 'No camera frame', (55, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                ok, enc = cv2.imencode('.jpg', blank, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                if ok:
                    frame = enc.tobytes()
                else:
                    time.sleep(0.08)
                    continue

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.04)

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/start_mission', methods=['POST'])
def api_start_mission():
    global mission_thread

    try:
        if not ros_ready:
            raise RuntimeError('ROS не готов')

        with mission_lock:
            if mission_status['running']:
                raise RuntimeError('Маршрут уже выполняется')

        data = request.get_json(force=True)
        route_ids = data.get('route', [])
        height = float(data.get('height', 1.5))
        speed = float(data.get('speed', 0.5))
        land_at_end = bool(data.get('land_at_end', False))

        if not route_ids:
            raise RuntimeError('Маршрут пустой')

        if height <= 0:
            raise RuntimeError('Высота должна быть больше 0')

        if speed <= 0:
            raise RuntimeError('Скорость должна быть больше 0')

        mission_stop_event.clear()

        mission_thread = threading.Thread(
            target=mission_worker,
            args=(route_ids, height, speed, land_at_end),
            daemon=True
        )
        mission_thread.start()

        return jsonify({
            'success': True,
            'message': 'Маршрут запущен'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/stop_mission', methods=['POST'])
def api_stop_mission():
    try:
        mission_stop_event.set()
        set_mission_status(
            running=False,
            message='Остановка маршрута запрошена',
            current_index=-1
        )
        return jsonify({
            'success': True,
            'message': 'Команда остановки отправлена'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/land', methods=['POST'])
def api_land():
    try:
        mission_stop_event.set()
        res = land()

        if not res.success:
            raise RuntimeError(res.message)

        set_mission_status(
            running=False,
            message='Посадка выполнена',
            current_index=-1
        )

        return jsonify({
            'success': True,
            'message': 'Посадка выполнена'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/emergency', methods=['POST'])
def api_emergency():
    try:
        data = request.get_json(force=True)
        mode = data.get('mode', 'land_now').strip()

        thread = threading.Thread(target=emergency_worker, args=(mode,), daemon=True)
        thread.start()

        return jsonify({
            'success': True,
            'message': 'Аварийное действие запущено'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/status')
def api_status():
    with mission_lock:
        return jsonify({
            'success': True,
            'ros_ready': ros_ready,
            'camera_topic': current_camera_topic,
            'camera_type': current_camera_type,
            'mission': mission_status
        })


if __name__ == '__main__':
    try:
        init_ros()
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True, use_reloader=False)
    except KeyboardInterrupt:
        print('\\nОстановка сервера по Ctrl+C')
    finally:
        try:
            mission_stop_event.set()
            if not rospy.is_shutdown():
                rospy.signal_shutdown('server stopped')
        except Exception:
            pass
