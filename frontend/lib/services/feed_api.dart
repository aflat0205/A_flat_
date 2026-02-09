import 'dart:convert';
import 'package:http/http.dart' as http;

class FeedApi {
  static const String baseUrl = "http://10.0.2.2:8000";

  static Future<Map<String, dynamic>> getFeed() async {
    final response = await http.get(Uri.parse("$baseUrl/v0/feed"));

    if (response.statusCode == 200) {
      return jsonDecode(response.body);
    } else {
      throw Exception("Failed to load feed");
    }
  }
}
