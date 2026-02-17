@app.post("/hightlights", response_model=ActionResponse)
def hightlights(game_state: FullGameState):
    global model, mappo, time_step
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    hightlights_agent_id = 0  # only for ai-1 hightlights for now
    topKQueue: TopKPriorityQueue = app.state.topKQueue
    trajBuffer: deque = app.state.trajBuffer
    lastSummaryTrajectory: SummaryTrajectory = app.state.lastSummaryTrajectory
    configs = app.state.hightlights_configs
    intervalSize = configs['intervalSize']
    statesAfter = configs['statesAfter']
    k = configs['k']
    l = configs['l']
    time_step += 1

    #img = render_grid(game_state.tiles, game_state.player, tile_size=32, width=game_state.gridSize[0], height=game_state.gridSize[1],
    #                  pov_agent_id=1, visibility_radius=1000)
    #display_grid_live(img)
    #raise HTTPException(status_code=400, detail="MAX_STEPS_REACHED")

    # terminating condition
    if time_step >= int(os.getenv("MAX_STEPS", "50")):
        save_topk_trajectories_to_file(topKQueue, mappo.args)
        print("Hightlights MAX_STEPS reached. Dumping topKQueue to file.")
        #import imageio
        #imageio.mimsave(os.path.join(mappo.args.data_path, "saliency_maps.gif"), app.state.saliency_list, fps=2)
        raise HTTPException(status_code=400, detail="MAX_STEPS_REACHED")

    feature_per_agent = app.state.feature_per_agent
    num_agents = len([player for player in game_state.player.values() if player.playerType == "agent"])
    #try:
    # iterate over all agents id agent-1, agent-2, ..., agent-n
    agent_ids = [player_id for player_id in game_state.player.keys() if game_state.player[player_id].playerType == "agent"]
    agent_ids.sort(key=lambda x: int(x.split('-')[1]))  # sort the agent ids to ensure consistent order
    
    observations_list = [
        # globalobsfeature has shape (n*101, )
        torch.FloatTensor(feature_per_agent[i].generate(game_state, player_id)).unsqueeze(0)  # (1, 202)
        for i, player_id in enumerate(agent_ids)
    ]  # list of (1, 202) tensors obs for all ai-agents in the game

    if len(observations_list) == 0:
        # log this to a file
        #raise HTTPException(status_code=400, detail="No AI agents found in the game state")
        return ActionResponse(actions={"0":6, "1":6})  # return dummy actions

    img_agent1 = render_grid(game_state.tiles, game_state.player, tile_size=32, width=game_state.gridSize[0], height=game_state.gridSize[1], pov_agent_id=1, visibility_radius=mappo.args.region_size)
    if len(observations_list) > 1:
        img_agent2 = render_grid(game_state.tiles, game_state.player, tile_size=32, width=game_state.gridSize[0], height=game_state.gridSize[1], pov_agent_id=2, visibility_radius=mappo.args.region_size)
    # bgr to rgb
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if "CNNAgent" in mappo.args.model_type:  # only CNNAgent can output saliency map
        saliency, s_grey = saliency_map(mappo.policy, observations_list[0], agent_id=0, game_state_img=img_agent1)  # visualize saliency map for the first agent
        #saliency = upscale_saliency(saliency, tiles_x=game_state.gridSize[0], tiles_y=game_state.gridSize[1], pixels_per_tile=32)
        saliency_uint8 = (saliency * 255).astype(np.uint8)
        # agent 2
        if len(observations_list) > 1:
            saliency2, s_grey2 = saliency_map(mappo.policy, observations_list[1], agent_id=1, game_state_img=img_agent2)  # visualize saliency map for the second agent
            #saliency2 = upscale_saliency(saliency2, tiles_x=game_state.gridSize[0], tiles_y=game_state.gridSize[1], pixels_per_tile=32)
            saliency2_uint8 = (saliency2 * 255).astype(np.uint8)
        img_vertical = np.concatenate((img_agent1, img_agent2), axis=0)
        saliency__grey_vertical = np.concatenate((s_grey, s_grey2), axis=0)

    # agent's posiiton
    agent_coords = []  # list of (x, y)
    for id in agent_ids:  # sorted upward
        player = game_state.player.get(id)
        agent_coords.append((player.x, player.y))

    overlayer = SaliencyOverlay(saliency_tile_size=16, img_tile_size=32, mean=True)
    clip_s_grey  = z_score_clip(s_grey, k=4)
    clip_s_grey2 = z_score_clip(s_grey2, k=4)
    overlay_img_a1 = overlayer.overlay_with_borders(img_agent1, clip_s_grey, top_k=1, border_color=(185, 239, 250), agent_coords=agent_coords[0])
    overlay_img_a2 = overlayer.overlay_with_borders(img_agent2, clip_s_grey2, top_k=1, border_color=(185, 250, 215), agent_coords=agent_coords[1])
    overlay_img = np.concatenate((overlay_img_a1, overlay_img_a2), axis=0)

    #plot_fig = plot_saliency_3d_surface(s_grey, game_state.RLTimeStep)
    #app.state.saliency_list.append(plot_fig)  # store the saliency map and plot for later visualization in the frontend
    display_grid_live(overlay_img)

    ## concatenate all observations into a (n_agents, n_agents*101) tensor
    obs_tensor = torch.cat(observations_list, dim=0).to(device)  # (n_agents, n_agents*101)

    with torch.no_grad():
        actions, logprob, entropy, values = mappo.act(obs_tensor)
    ## Convert actions to dictionary
    #action_dict = {str(i): int(actions[i]) for i in range(len(actions))}
    #return ActionResponse(actions=action_dict)

    # check if there is gamestate.screenshot for the hightlights_agent_id
    #if game_state.gameScreenshot == "":
    #    print("No screenshot found in game state for hightlights. Skipping.")
    #    return ActionResponse(actions={str(i): int(actions[i]) for i in range(len(actions))})

    I = compute_importance(model, obs_tensor, entropy, hightlights_agent_id)  # line 14 of pseudocode

    # check if eat has gradient
    trajBuffer.append((obs_tensor.detach().cpu().numpy()[hightlights_agent_id], actions.cpu().numpy(), game_state.gameScreenshot,
                    I, overlay_img))  # line 11 of pseudocode
    with app.state.hightlights_lock:                                      # line 12 of pseudocode
        if app.state.hightlights_counter > 0:
            app.state.hightlights_counter -= 1
    

    if intervalSize - app.state.hightlights_counter == statesAfter:  # line 15 of pseudocode
        print(f"Setting lastSummaryTrajectory with trajBuffer of length {len(trajBuffer)}")
        lastSummaryTrajectory.setTrajectory(trajBuffer)

    # check if heap size < k or I > min importance in heap
    # by defauilt, heapq is min-heap, but we store (-importance, trajectory) to make it max-heap  
    if (len(topKQueue.heap) < k or I > topKQueue.maxImportance()) and app.state.hightlights_counter == 0:
        if topKQueue.__len__() >= k:
            removed = heapq.heappop(topKQueue.heap)  # remove the max entropy trajectory
            print(f"Removed trajectory with importance {-removed[0]} from topKQueue. Current size: {len(topKQueue)}")
        # create a SummaryTrajectory object
        new_summary_traj = SummaryTrajectory(I)
        topKQueue.add(new_summary_traj)
        print(f"Added new trajectory with importance {I} to topKQueue. Current size: {len(topKQueue)}")
        app.state.hightlights_counter = intervalSize  # reset counter
        app.state.lastSummaryTrajectory = new_summary_traj
        print(f"time step: {game_state.RLTimeStep}")


    # Convert actions to dictionary
    action_dict = {str(i): int(actions[i]) for i in range(len(actions))}

    if time_step % 100 == 0:
        save_topk_trajectories_to_file(topKQueue, mappo.args)
    # Return the top-k trajectories as part of the response
    return ActionResponse(actions=action_dict)

