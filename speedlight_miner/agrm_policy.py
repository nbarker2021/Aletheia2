class AGRMPolicy:
    def __init__(self, Kx=(1024,4096,8192), Km=(64,256,1024), Kv=(16,64,128)):
        self.Kx_choices=Kx; self.Km_choices=Km; self.Kv_choices=Kv
    def choose(self, reuse_R: float, Q_total: float):
        if Q_total >= 0.4:
            policy=1; kx=max(self.Kx_choices); km=min(self.Km_choices); kv=min(self.Kv_choices)
        else:
            policy=0; km=self.Km_choices[1 if reuse_R<8 else 2]; kv=self.Kv_choices[1]; kx=self.Kx_choices[1]
        return {"policy":policy,"Kx":kx,"Km":km,"Kv":kv}
