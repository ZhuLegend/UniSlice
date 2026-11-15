//! comm单元测试模块

#[tokio::test]
/// 测试获取未使用端口
async fn test_get_unused_port() {
    let port = crate::get_unused_port();
    assert!(port > 0);
}
