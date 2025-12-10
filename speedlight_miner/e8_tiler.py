class E8Tiler:
    @staticmethod
    def key(nonce:int, extranonce:int, merkle_class:int, version:int, timestamp:int)->int:
        x=(nonce*0x9E3779B185EBCA87)^(extranonce<<1)^(merkle_class<<3)^(version<<5)^(timestamp<<7)
        return x & ((1<<64)-1)
    @staticmethod
    def owns(agent_id:int, agent_count:int, key:int)->bool:
        return (key % max(1,agent_count)) == agent_id
