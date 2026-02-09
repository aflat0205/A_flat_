import 'dart:math' as math;
import 'dart:ui';

import 'package:flutter/material.dart';
import 'package:video_player/video_player.dart';

import '../services/feed_api.dart';


class FeedScreen extends StatefulWidget {
  const FeedScreen({super.key});

  @override
  State<FeedScreen> createState() => _FeedScreenState();
}


class _FeedScreenState extends State<FeedScreen> {
  // static const double _bottomBarHeight = 64.0;

  final PageController _pageController = PageController();
  final List<Map<String, dynamic>> _items = [];
  final Map<int, VideoPlayerController> _controllers = {};

  int _currentIndex = 0;
  bool _loadingMore = false;

  int _selectedTab = 0;

  @override
  void initState() {
    super.initState();
    _prime();
  }

  Future<void> _prime() async {
    await _ensureItems(3);
    await _prepareController(0);
    await _prepareController(1);
    _playOnly(_currentIndex);
    if (mounted) setState(() {});
  }

  Future<void> _ensureItems(int minCount) async {
    if (_loadingMore) return;
    _loadingMore = true;
    try {
      while (_items.length < minCount) {
        final item = await FeedApi.getFeed();
        _items.add(item);
      }
    } finally {
      _loadingMore = false;
    }
  }

  Future<void> _prepareController(int index) async {
    if (index < 0) return;
    if (index >= _items.length) return;
    if (_controllers.containsKey(index)) return;

    final url = _items[index]["video_url"]?.toString();
    if (url == null || url.isEmpty) return;

    final c = VideoPlayerController.networkUrl(Uri.parse(url));
    _controllers[index] = c;

    try {
      await c.initialize();
      c.setLooping(true);
    } catch (_) {
      _controllers.remove(index);
      await c.dispose();
    }
  }

  void _playOnly(int index) {
    for (final entry in _controllers.entries) {
      final i = entry.key;
      final c = entry.value;
      if (!c.value.isInitialized) continue;

      if (i == index) {
        c.play();
      } else {
        c.pause();
      }
    }
  }

  Future<void> _onPageChanged(int index) async {
    _currentIndex = index;

    if (_items.length - index <= 2) {
      await _ensureItems(_items.length + 3);
    }

    await _prepareController(index);
    await _prepareController(index + 1);

    _cleanupControllers(keepAround: index);
    _playOnly(index);

    if (mounted) setState(() {});
  }

  void _cleanupControllers({required int keepAround}) {
    final keys = _controllers.keys.toList();
    for (final k in keys) {
      if ((k - keepAround).abs() > 1) {
        final c = _controllers.remove(k);
        if (c != null) {
          c.pause();
          c.dispose();
        }
      }
    }
  }

  @override
  void dispose() {
    _pageController.dispose();
    for (final c in _controllers.values) {
      c.dispose();
    }
    super.dispose();
  }

  Widget _bottomBar() {
    final bottomInset = MediaQuery.of(context).padding.bottom;

    return Container(
      color: Colors.black,
      padding: EdgeInsets.only(
        left: 18,
        right: 18,
        top: 18,
        bottom: bottomInset + 14,
      ),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          _navItem(icon: Icons.home_filled, label: "Home", index: 0),
          _navItem(icon: Icons.search, label: "Discover", index: 1),
          _plusButton(),
          _navItem(icon: Icons.inbox, label: "Inbox", index: 3),
          _navItem(icon: Icons.person, label: "Profile", index: 4),
        ],
      ),
    );
  }

  Widget _navItem({required IconData icon, required String label, required int index}) {
    final active = _selectedTab == index;

    return InkWell(
      onTap: () => setState(() => _selectedTab = index),
      child: Icon(
        icon,
        size: 26,
        color: active ? Colors.white : Colors.white54,
      ),
    );
  }

  Widget _plusButton() {
    return InkWell(
      onTap: () {
      },
      child: Container(
        width: 46,
        height: 32,
        decoration: BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.circular(10),
        ),
        child: const Icon(Icons.add, color: Colors.black, size: 24),
      ),
    );
  }

  Widget _videoWithSmartFit({
    required VideoPlayerController c,
    required double bodyWidth,
    required double bodyHeight,
  }) {
    final videoAR = c.value.aspectRatio;
    final screenAR = bodyWidth / bodyHeight;

    final crop = _coverCropFraction(videoAR: videoAR, screenAR: screenAR);

    const cropThreshold = 0.18;
    final useBlurContain = crop >= cropThreshold;

    if (!useBlurContain) {
      return FittedBox(
        fit: BoxFit.cover,
        child: SizedBox(
          width: c.value.size.width,
          height: c.value.size.height,
          child: VideoPlayer(c),
        ),
      );
    }

    return Stack(
      fit: StackFit.expand,
      children: [
        FittedBox(
          fit: BoxFit.cover,
          child: SizedBox(
            width: c.value.size.width,
            height: c.value.size.height,
            child: VideoPlayer(c),
          ),
        ),
        Positioned.fill(
          child: ImageFiltered(
            imageFilter: ImageFilter.blur(sigmaX: 18, sigmaY: 18),
            child: Container(color: Colors.black.withOpacity(0.35)),
          ),
        ),
        Center(
          child: AspectRatio(
            aspectRatio: videoAR,
            child: VideoPlayer(c),
          ),
        ),
      ],
    );
  }

  double _coverCropFraction({required double videoAR, required double screenAR}) {
    if ((videoAR - screenAR).abs() < 1e-6) return 0.0;

    final screenW = screenAR;
    final screenH = 1.0;

    final videoW = videoAR;
    final videoH = 1.0;

    final scale = math.max(screenW / videoW, screenH / videoH);

    final scaledW = videoW * scale;
    final scaledH = videoH * scale;

    final cropW = math.max(0.0, 1.0 - (screenW / scaledW));
    final cropH = math.max(0.0, 1.0 - (screenH / scaledH));

    return math.max(cropW, cropH);
  }

  Widget _buildPage({
    required int index,
    required double bodyWidth,
    required double bodyHeight,
  }) {
    if (index >= _items.length) {
      return const Center(child: CircularProgressIndicator());
    }

    final item = _items[index];
    final id = item["content_id"]?.toString() ?? "?";
    final score = item["score"]?.toString() ?? "-";
    final c = _controllers[index];

    return Stack(
      fit: StackFit.expand,
      children: [
        if (c != null && c.value.isInitialized)
          _videoWithSmartFit(c: c, bodyWidth: bodyWidth, bodyHeight: bodyHeight)
        else
          const Center(child: CircularProgressIndicator()),

        Positioned(
          left: 16,
          bottom: 24,
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(id, style: const TextStyle(color: Colors.white, fontSize: 28)),
              const SizedBox(height: 6),
              Text("score: $score", style: const TextStyle(color: Colors.white70)),
            ],
          ),
        ),
      ],
    );
  }

  @override
  Widget build(BuildContext context) {
    if (_items.isEmpty) {
      return const Scaffold(
        backgroundColor: Colors.black,
        body: Center(child: CircularProgressIndicator()),
      );
    }

    return Scaffold(
      backgroundColor: Colors.black,

      body: LayoutBuilder(
        builder: (context, constraints) {
          final bodyWidth = constraints.maxWidth;
          final bodyHeight = constraints.maxHeight;

          return PageView.builder(
            controller: _pageController,
            scrollDirection: Axis.vertical,
            onPageChanged: _onPageChanged,
            itemBuilder: (context, index) {
              return _buildPage(index: index, bodyWidth: bodyWidth, bodyHeight: bodyHeight);
            },
          );
        },
      ),

      bottomNavigationBar: _bottomBar(),
    );
  }
}
