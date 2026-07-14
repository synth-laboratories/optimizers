use crate::candidate::MapoCandidate;

pub fn propose_candidates(
    parent: &MapoCandidate,
    generation: usize,
    proposals_per_generation: usize,
) -> Vec<MapoCandidate> {
    (0..proposals_per_generation)
        .map(|index| deterministic_grid_candidate(parent, generation, index))
        .collect()
}

fn deterministic_grid_candidate(
    parent: &MapoCandidate,
    generation: usize,
    index: usize,
) -> MapoCandidate {
    let mut candidate = parent.clone();
    candidate.id = format!("mapo_g{generation}_p{index}");
    candidate.generation = generation;
    candidate.parent_id = Some(parent.id.clone());
    candidate.train_score = None;
    candidate.heldout_score = None;
    candidate.selection_score = None;
    candidate.roles.clear();

    match index % 8 {
        0 => {
            candidate.protocol.mode = "pure_decentralized".to_string();
            candidate.protocol.max_chars = 220;
            candidate.roles.insert(
                "default".to_string(),
                "Use symbolic.progress_hints as the action shortlist. In squad runs, send one compact party message when lane assignment, objective carrier, blocked route, clue/counterplay result, or extraction regroup changes. Resolve visible counterplay or clue objects before picking up the objective unless the route is blocked. After the objective is held, prioritize returning to the escape tile over extra combat or treasure.".to_string(),
            );
            candidate.roles.insert(
                "dgp_hero_0".to_string(),
                "Squad coordinator: send the opening lane assignment, then guard the carrier route. Message only new blocked routes, carrier status, or regroup point.".to_string(),
            );
            candidate.roles.insert(
                "dgp_hero_1".to_string(),
                "Scout specialist: check side lanes and clue/counterplay objects. Message concise findings with room and next action, then rejoin the objective route.".to_string(),
            );
            candidate.roles.insert(
                "dgp_hero_2".to_string(),
                "Door and chokepoint guard: open/hold routes, clear only blocking enemies, and message when a lane is safe or blocked.".to_string(),
            );
            candidate.roles.insert(
                "dgp_hero_3".to_string(),
                "Escort support: stay near the objective carrier or escape lane. Message when escort, healing, or regroup is needed.".to_string(),
            );
            candidate.roles.insert(
                "barbarian".to_string(),
                "Frontline rule: open and hold doors, guard the carrier, clear only blocking enemies, and message the exact blocked door or threat. Do not wander into a second fork while the wizard is resolving a clue.".to_string(),
            );
            candidate.roles.insert(
                "wizard".to_string(),
                "Specialist rule: inspect clues, runes, plaques, ledgers, and counterplay objects before the item moves. Message the clue result in one sentence, then support the carrier back to escape.".to_string(),
            );
        }
        1 => {
            candidate.protocol.mode = "situational_lead_taking".to_string();
            candidate.protocol.max_chars = 240;
            candidate.protocol.leader_policy = "first_hero".to_string();
            candidate.protocol.followers_can_reply = true;
            candidate.roles.insert(
                "default".to_string(),
                "Run a squad plan: coordinator assigns lanes, scout resolves clues/counterplay, guard keeps chokepoints open, support escorts the carrier. Message only plan, discovery, carrier, blocked route, handoff, or regroup point. Prefer legal progress_hint actions and avoid low-value attacks when movement, search, interact, or objective actions are legal.".to_string(),
            );
            candidate.roles.insert(
                "dgp_hero_0".to_string(),
                "Take coordinator/guard responsibility. On the first legal party-message opportunity, send one compact plan assigning scout, guard, support, and carrier route. Then guard the objective route and update only blocked route or carrier changes.".to_string(),
            );
            candidate.roles.insert(
                "dgp_hero_1".to_string(),
                "Take scout/specialist responsibility. Search or interact with clue and counterplay objects, send one compact discovery/counterplay result to the party, then regroup rather than opening a distant second front.".to_string(),
            );
            candidate.roles.insert(
                "dgp_hero_2".to_string(),
                "Take chokepoint/door responsibility. Open safe lanes, hold one-tile blockers, avoid blocking the scout, and message only lane clear/blocked status.".to_string(),
            );
            candidate.roles.insert(
                "dgp_hero_3".to_string(),
                "Take support/escort responsibility. Stay close enough to reinforce carrier or scout, handle adjacent support actions, and message regroup/escort needs.".to_string(),
            );
        }
        2 => {
            candidate.protocol.mode = "pure_decentralized".to_string();
            candidate.protocol.max_chars = 140;
            candidate.roles.insert(
                "default".to_string(),
                "Message exactly when entering a fork, finding a clue/counterplay object, seeing the objective, carrying the objective, or finding a blocked route. Use terse status: role, room, action, next tile. Otherwise take the best non-message progress action from symbolic.progress_hints or legal_actions. Once the objective is recovered or carried, stop optional search/combat and move toward the escape tile unless an enemy or door directly blocks that route.".to_string(),
            );
        }
        3 => {
            candidate.protocol.mode = "no_message".to_string();
            candidate.protocol.max_chars = 32;
            candidate.roles.insert(
                "default".to_string(),
                "Avoid messages. Use symbolic.progress_hints when present. Prefer search/interact/objective/move actions over attacks unless an enemy blocks the route. If carrying the objective, move toward escape immediately.".to_string(),
            );
        }
        4 => {
            candidate.protocol.mode = "pure_decentralized".to_string();
            candidate.protocol.max_chars = 160;
            candidate.roles.insert(
                "default".to_string(),
                "Use relay discipline. Send compact tactical messages only when split-party state changes: lane owner, location, objective carrier, blocked route, clue result, counterplay status, enemy threat, or extraction regroup. If no such state changed, choose a physical progress action.".to_string(),
            );
            candidate.roles.insert(
                "dgp_hero_0".to_string(),
                "Relay lead: maintain the shared route picture. Message lane ownership and regroup point, then move or guard rather than repeating chat.".to_string(),
            );
            candidate.roles.insert(
                "dgp_hero_1".to_string(),
                "Forward scout: report only new rooms, objective/clue sightings, and blocked lanes. Otherwise keep revealing the map.".to_string(),
            );
            candidate.roles.insert(
                "dgp_hero_2".to_string(),
                "Rear guard: report threats that can cut off escape, then clear or hold the chokepoint.".to_string(),
            );
            candidate.roles.insert(
                "dgp_hero_3".to_string(),
                "Carrier support: report when the carrier route changes, then escort or prepare extraction.".to_string(),
            );
        }
        5 => {
            candidate.protocol.mode = "master_to_slaves".to_string();
            candidate.protocol.max_chars = 240;
            candidate.protocol.leader_policy = "role".to_string();
            candidate.protocol.leader_role = "barbarian".to_string();
            candidate.protocol.followers_can_reply = true;
            candidate.roles.insert(
                "dgp_hero_0".to_string(),
                "Act as field commander. Assign hero_1 to scout clues, hero_2 to hold doors, hero_3 to escort carrier. Message only route assignments, blocked lanes, and extraction orders.".to_string(),
            );
            candidate.roles.insert(
                "dgp_hero_1".to_string(),
                "Scout under commander intent. Prioritize search/interact on clues and counterplay, then report only concrete findings or blocked paths.".to_string(),
            );
            candidate.roles.insert(
                "dgp_hero_2".to_string(),
                "Door guard under commander intent. Hold chokepoints, avoid blocking scouts, clear only route-blocking enemies, and report lane state.".to_string(),
            );
            candidate.roles.insert(
                "dgp_hero_3".to_string(),
                "Escort under commander intent. Stay near the carrier or escape route, support adjacent heroes, and report when regroup is needed.".to_string(),
            );
            candidate.roles.insert(
                "barbarian".to_string(),
                "Act as field lead. Assign route ownership, keep the return lane open, ask the wizard for clue/counterplay handling, and escort the objective carrier. Message only concrete route, guard, or carrier instructions.".to_string(),
            );
            candidate.roles.insert(
                "wizard".to_string(),
                "Follow the field lead unless a clue, rune, plaque, ledger, or counterplay object is visible; then handle it and report the result before the objective is moved.".to_string(),
            );
        }
        6 => {
            candidate.protocol.mode = "situational_lead_taking".to_string();
            candidate.protocol.max_chars = 240;
            candidate.protocol.leader_policy = "first_hero".to_string();
            candidate.protocol.followers_can_reply = true;
            candidate.roles.insert(
                "default".to_string(),
                "Use baton leadership. Current lead messages only when assigning a new lane, reporting a clue/counterplay result, naming the carrier, or handing off to the hero best positioned for the next room. If you hand off leadership, include handoff_lead_to and a concrete reason. Otherwise prefer physical progress over chat.".to_string(),
            );
            candidate.roles.insert(
                "dgp_hero_0".to_string(),
                "Start as lead and assign squad lanes. Hand off to scout for clues, guard for chokepoints, or support for extraction when their role becomes pivotal.".to_string(),
            );
            candidate.roles.insert(
                "dgp_hero_1".to_string(),
                "Accept leadership when scouting/clues become pivotal. Report the finding and hand leadership back toward carrier/extraction.".to_string(),
            );
            candidate.roles.insert(
                "dgp_hero_2".to_string(),
                "Accept leadership when a door, enemy, or blocked lane controls progress. Report clear/blocked status, then hand off.".to_string(),
            );
            candidate.roles.insert(
                "dgp_hero_3".to_string(),
                "Accept leadership when the objective is carried or extraction is near. Report regroup point and escort route.".to_string(),
            );
        }
        _ => {
            candidate.protocol.mode = "pure_decentralized".to_string();
            candidate.protocol.max_chars = 220;
            candidate.roles.insert(
                "default".to_string(),
                "Optimize for squad completion: reveal rooms quickly, use interact/search actions on named objects, keep one hero near the escape route, pick up the objective only after a route back is known, and escape once carried. Send at most one short squad status message per hero when your role changes or you find objective/clue/blocked route.".to_string(),
            );
            candidate.roles.insert(
                "dgp_hero_0".to_string(),
                "Route captain: message one initial plan, then convert progress through doors, objective, and escape.".to_string(),
            );
            candidate.roles.insert(
                "dgp_hero_1".to_string(),
                "Fast scout: message only discoveries that change the plan, then keep moving."
                    .to_string(),
            );
            candidate.roles.insert(
                "dgp_hero_2".to_string(),
                "Guard: message only when a route is safe or blocked, then act physically."
                    .to_string(),
            );
            candidate.roles.insert(
                "dgp_hero_3".to_string(),
                "Escort: message only carrier/extraction status, then stay useful near the objective route.".to_string(),
            );
        }
    }
    candidate
}
