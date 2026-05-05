from __future__ import annotations

from agent import HybridMemoryAgent


def seed(agent: HybridMemoryAgent) -> None:
    memories = [
        "Tôi đã đọc một ghi chú về Kubernetes autoscaling: HPA tăng số pod theo CPU, còn cluster autoscaler tăng node khi thiếu tài nguyên.",
        "Cloud security checklist gồm IAM least privilege, secret rotation, network policy, audit log và mã hóa dữ liệu ở trạng thái nghỉ.",
        "Một bài về tự động mở rộng hạ tầng giải thích cách scale theo lưu lượng người dùng và giữ latency ổn định sau warmup.",
        "Tôi thích tài liệu tiếng Việt ngắn, có ví dụ thực tế, đặc biệt về cloud, DevOps và AI platform.",
        "Ghi chú gần đây: hybrid search kết hợp BM25 với vector search bằng RRF để xử lý query mixed Việt Anh tốt hơn.",
    ]
    for memory in memories:
        agent.remember(memory)


def main() -> int:
    agent = HybridMemoryAgent()
    seed(agent)

    queries = [
        "Tôi đã đọc gì về Kubernetes?",
        "Recommend đọc gì tiếp",
        "Tôi đang quan tâm gì gần đây?",
        "Tài liệu về tự động mở rộng hạ tầng?",
        "Cho tôi summary cloud security",
    ]
    for i, query in enumerate(queries, 1):
        print("=" * 78)
        print(f"Query {i}: {query}")
        print(agent.recall(query))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
