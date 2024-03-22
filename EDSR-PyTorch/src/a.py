path = r'/home/srteam/lrq/EDSR-PyTorch/experiment/{}/test{}.txt'.format(self.args.data_test[0], self.scale[0])
if self.args.save_results: self.ckp.begin_background()  ######
# t0 = time.time() #resd start
for idx_data, d in enumerate(self.loader_test):
    for idx_scale, scale in enumerate(self.scale):  # 0,4
        d.dataset.set_scale(idx_scale)
        for lr, hr, filename in tqdm(d, ncols=80):  # 创建一个进度条来跟踪迭代的进度
            t4 = time.time()  #######
            lr, hr = self.prepare(lr, hr)  # tensor().cuda
            diff = time.time() - t4  ##########
            diffg += diff
            self.ckp.write_log('cpu->gpu: {:.3f}s\n'.format(diffg))

            num += 1


            if num <= 30 and num % skip == 0:
                sr = self.model(lr, idx_scale)  # tensor().cuda (1,3,2160,3840)
                self.ckp.write_log('sr num is :{:01d}\n'.format(num))

            bic = core.imresize(lr.squeeze(0), self.scale[0])
            # self.ckp.write_log('bicubic num is :{:01d}\n'.format(num))

            gpu_id = 0  # GPU ID
            if num=0:
                for i in range(3):
                    encFrame2 = np.ndarray(shape=(0), dtype=np.uint8)  ###
                    sr = sr.squeeze(0).contiguous()  ######torch.Size([3, 2160, 3840]
                    sr_surface = to_nv12.run(SamplePyTorch.tensor_to_surface(sr, gpu_id))
                    # Encoded video frame
                    success_sr = nvEnc2.EncodeSingleSurface(sr_surface, encFrame2, sync=False)  # False，则编码操作是异步的

                    encFrame = np.ndarray(shape=(0), dtype=np.uint8)  ###
                    bic = bic.squeeze(0).contiguous()
                    bic_surface = to_nv12.run(SamplePyTorch.tensor_to_surface(bic, gpu_id))
                    success_bic = nvEnc.EncodeSingleSurface(bic_surface, encFrame, sync=False)  # False，则编码操作是异步的

            if num % skip == 0:
                encFrame2 = np.ndarray(shape=(0), dtype=np.uint8)  ###
                sr = sr.squeeze(0).contiguous()  ######torch.Size([3, 2160, 3840]
                sr_surface = to_nv12.run(SamplePyTorch.tensor_to_surface(sr, gpu_id))
                # Encoded video frame
                success_sr = nvEnc2.EncodeSingleSurface(sr_surface, encFrame2, sync=False)  # False，则编码操作是异步的

            encFrame = np.ndarray(shape=(0), dtype=np.uint8)  ###
            bic = bic.squeeze(0).contiguous()
            bic_surface = to_nv12.run(SamplePyTorch.tensor_to_surface(bic, gpu_id))
            success_bic = nvEnc.EncodeSingleSurface(bic_surface, encFrame, sync=False)  # False，则编码操作是异步的
            # trans: 0.413s

            # print(encFrame.device)
            if num % skip == 0:
                if success_sr:
                    print("sr:", str(num), str(encFrame2.shape))  # byteArray
                    pkt.append(encFrame2)  # .copy()
                    # print(pkt)
                    framesReceived += 1
                else:
                    pkt.append(0)
                    N_sr.append(num)

            if success_bic:
                if num % skip != 0:
                    print("bic:", str(num), str(encFrame.shape))
                    pkt.append(encFrame.copy())
                    # print(pkt)
                    framesReceived += 1
            else:
                pkt.append(0)
                N_bic.append(num)



        j=-1
        while True:
            #print("\n SR " + str(j) + str(N_sr[j]))
            j=j+1
            success = nvEnc2.FlushSinglePacket(encFrame2)
            if success  and j < len(N_sr):
                if N_sr[j] %skip ==0:
                    f = encFrame2.copy()
                    print(encFrame2,f)
                    pkt[N_sr[j]] = encFrame2.copy()
                    framesReceived += 1
                    framesFlushed += 1
                    print("\nSR fush "+str(j)+str(N_sr[j])+str(f)+'\n')
                    #print(pkt)
            else:
                break
        j=-1
        while True:
                #print("\n  bic  " + str(j) + str(N_bic[j]))
                j = j + 1
                success = nvEnc.FlushSinglePacket(encFrame)
                if success and j < len(N_bic):
                    if N_bic[j] % skip != 0:
                        f = encFrame.copy()
                        pkt[N_bic[j]] = encFrame.copy()
                        framesReceived += 1
                        framesFlushed += 1
                        print("\nbic fush " + str(j) + str(N_bic[j]) + str(f) + '\n')
                        #print(pkt)

                else:
                    break