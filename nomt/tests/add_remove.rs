mod common;

use common::Test;
use hex_literal::hex;
use nomt::trie::Node;

#[test]
fn add_remove_1000() {
    let mut accounts = 0;
    let mut t = Test::new("add_remove");

    // These fixtures track the current `common::account_path` distribution.
    let expected_roots = [
        hex!("0000000000000000000000000000000000000000000000000000000000000000"),
        hex!("18a457ed03d9d28f8bf84aa05dc3c8005e35564e91d05bf6131094d4f3398528"),
        hex!("51c06663bc9f0e754ada8512e3b4f97958007cd37087084ac3b80417e6b7d0be"),
        hex!("4ea35ff714b076a9467442f65e21ec888651f5120966f2681393e63dfd12645a"),
        hex!("38e03f8eda1c1d47cad38420bcf467cd9bb08a8be3a81606194db53a94e988ab"),
        hex!("7849ef9330e40dbe28b2005208c5af476ec43d10695df9b21abeec9df2194667"),
        hex!("6d1de05e86b0448033a7a8a0d500fcb381396bfded615670b423defad34ce5a8"),
        hex!("12c9d4879de20229f771bd62d92b860ee1f18cbef04b985625c7a473a6ef708e"),
        hex!("74b43826e82d1c1ee9f7cd1e260bfdedb9db55504a34e7ddb86552fad5d11f8b"),
        hex!("4afa73ab74926485eb4322b9c7f58cb70085b17c20cdc31f35017af9ee45a019"),
        hex!("542b4aae8ba2e202184b04becaed29db107680cf3c58b54d7f2ff615bfbce458"),
    ];

    let mut root = Node::default();
    for i in 0..10 {
        let _ = t.read_id(0);
        for _ in 0..100 {
            common::set_balance(&mut t, accounts, 1000);
            accounts += 1;
        }
        {
            root = t.commit().0.into_inner();
        }

        assert_eq!(root, common::expected_root(accounts));
        assert_eq!(root, expected_roots[i + 1]);
    }

    assert_eq!(root, expected_roots[10]);

    for i in 0..10 {
        for _ in 0..100 {
            accounts -= 1;
            common::kill(&mut t, accounts);
        }
        {
            root = t.commit().0.into_inner();
        }

        assert_eq!(root, common::expected_root(accounts));
        assert_eq!(root, expected_roots[10 - i - 1]);
    }
}

#[test]
fn delete_missing_key_in_empty_db() {
    let mut t = Test::new("delete_missing_key_in_empty_db");
    t.write([0; 32], None);
    assert_eq!(t.commit().0.into_inner(), Node::default());
}

#[test]
fn delete_missing_key_in_non_empty_db() {
    let mut t = Test::new("delete_missing_key_in_non_empty_db");
    common::set_balance(&mut t, 0, 1000);
    let root = t.commit().0;
    common::kill(&mut t, 1);
    assert_eq!(t.commit().0, root);
}
